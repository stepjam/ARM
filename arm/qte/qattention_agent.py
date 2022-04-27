import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary

from arm import utils
from arm.utils import visualise_voxel, stack_on_channel
from arm.c2farm.voxel_grid import VoxelGrid

NAME = 'QAttentionAgent'
REPLAY_BETA = 1.0


class QFunction(nn.Module):

    def __init__(self,
                 unet_3d: nn.Module,
                 voxel_grid: VoxelGrid,
                 bounds_offset: float,
                 rotation_resolution: float,
                 device):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxel_grid = voxel_grid
        self._bounds_offset = bounds_offset
        self._qnet = copy.deepcopy(unet_3d)
        self._qnet._dev = device
        self._qnet.build()

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
        return coords, rot_and_grip_indicies

    def __argmax_3d(self, tensor_orig, topk: int = 1):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        # idxs = tensor_orig.view(b, c, -1).argmax(-1)
        values, idxs = tensor_orig.view(b, -1, 1).topk(topk, 1)  # (B, 1, K)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], -1)  # (B, 1, K, 3)
        return indices, values

    def choose_highest_trans_action(self, q_trans, topk: int = 1):
        coords, values = self.__argmax_3d(q_trans, topk)
        return coords, values

    def choose_highest_rot_grip_action(self, q_rot_grip):
        rot_and_grip_indicies = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
        return rot_and_grip_indicies

    def forward(self, x: list, proprio, pcd: list,
                bounds=None, latent=None):
        # x will be list of list (list of [rgb, pcd])
        b, t, p, w, h = pcd[0].shape

        pcd_flat = torch.cat(
            [pc.view(b * t, p, w, h).permute(0, 2, 3, 1).reshape(b * t, -1, 3) for pc in pcd], 1)

        image_features = [xx[0].view(b * t, -1, w, h) for xx in x]
        feat_size = image_features[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b * t, -1, feat_size) for p in
             image_features], 1)

        voxel_grid = self._voxel_grid.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)

        # Swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()
        voxel_grid = voxel_grid.view((b, t) + voxel_grid.shape[1:])

        q_trans, rot_and_grip_q = self._qnet(voxel_grid, proprio, latent)
        return q_trans, rot_and_grip_q, voxel_grid

    def latents(self):
        return self._qnet.latent_dict


class QAttentionAgent(Agent):

    def __init__(self,
                 layer: int,
                 coordinate_bounds: list,
                 unet3d: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 timesteps: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 exploration_strategy: str,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 nstep: int = 1,
                 lr: float = 0.0001,
                 lambda_trans_qreg: float = 1e-6,
                 lambda_rot_qreg: float = 1e-6,
                 grad_clip: float = 20.,
                 include_low_dim_state: bool = False,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 tree_search_breadth: int = 10,
                 tree_during_update: bool = True,
                 tree_during_act: bool = True
                 ):
        self._layer = layer
        self._lambda_trans_qreg = lambda_trans_qreg
        self._lambda_rot_qreg = lambda_rot_qreg
        self._coordinate_bounds = coordinate_bounds
        self._unet3d = unet3d
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._tau = tau
        self._gamma = gamma
        self._nstep = nstep
        self._lr = lr
        self._grad_clip = grad_clip
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._timesteps = timesteps
        self._exploration_strategy = exploration_strategy
        self._lambda_weight_l2 = lambda_weight_l2

        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution

        self._name = NAME + '_layer' + str(self._layer)

        self._tree_search_breadth = tree_search_breadth
        self._tree_during_update = tree_during_update
        self._tree_during_act = tree_during_act
        self._next_depth_qattention = None

    def give_next_layer_qattention(self, qattention: 'QAttentionAgent'):
        self._next_depth_qattention = qattention

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        vox_grid = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=(self._batch_size if training else 1) * self._timesteps,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )
        self._vox_grid = vox_grid

        self._q = QFunction(self._unet3d, vox_grid, self._bounds_offset,
                            self._rotation_resolution,
                            device).to(device).train(training)
        self._q_target = None
        if training:
            self._q_target = QFunction(self._unet3d, vox_grid,
                                       self._bounds_offset,
                                       self._rotation_resolution,
                                       device).to(
                device).train(False)
            for param in self._q_target.parameters():
                param.requires_grad = False
            utils.soft_updates(self._q, self._q_target, 1.0)
            self._optimizer = torch.optim.Adam(
                self._q.parameters(), lr=self._lr,
                weight_decay=self._lambda_weight_l2)

            logging.info('# Q Params: %d' % sum(
                p.numel() for p in self._q.parameters() if p.requires_grad))
        else:
            for param in self._q.parameters():
                param.requires_grad = False

        grid_for_crop = torch.arange(
            0, self._image_crop_size, device=device).unsqueeze(0).repeat(
            self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        self._device = device


    def get_highest_tree_value(self, obs, proprio, pcd, coords, prev_vox, index_target: bool = False):
        k = 1 if self._next_depth_qattention is None else self._tree_search_breadth

        bounds = self._coordinate_bounds.repeat(self._timesteps, 1)
        if coords is not None:
            bounds = torch.cat(
                [coords - self._bounds_offset, coords + self._bounds_offset], dim=1)

        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size

        q, q_rot_grip, voxel_grid = self._q(obs, proprio, pcd, bounds, prev_vox)
        coord_idxs_d, coord_values_d = self._q.choose_highest_trans_action(q, k)  # (B, K, ...) for coords
        rot_grip_index = self._q.choose_highest_rot_grip_action(q_rot_grip)

        if index_target:
            q_targ, q_rot_grip_targ, voxel_grid_targ = self._q_target(obs, proprio, pcd, bounds, prev_vox)
            b, k, _ = coord_idxs_d.shape
            _, c, d, h, w = q_targ.shape
            flat_coord_idxs_d = coord_idxs_d.view(b*k, 3)
            coord_values_d = self._get_value_from_voxel_index(q_targ.unsqueeze(1).repeat(1, k, 1, 1, 1, 1).view(b * k, c, d, h, w), flat_coord_idxs_d)
            coord_values_d = coord_values_d.view(b, k, -1)
            if rot_grip_index is not None:
                q_rot_grip = self._get_value_from_rot_and_grip(q_rot_grip_targ, rot_grip_index)

        coord_idxs_d = coord_idxs_d.int()

        if self._next_depth_qattention is not None:
            coords_dp1_, coord_values_dp1_ = [], []
            for i in range(k):
                coords_d = bounds[:, :3] + res * coord_idxs_d[:, i] + res / 2
                coords_dp1, coord_values_dp1, _, _, _, _ = self._next_depth_qattention.get_highest_tree_value(obs, proprio, pcd, coords_d, voxel_grid, index_target)  # will be (B, 1, 1)
                coords_dp1_.append(coords_dp1)
                coord_values_dp1_.append(coord_values_dp1)
            coords_dp1 = torch.cat(coords_dp1_, 1)
            coord_values_dp1 = torch.cat(coord_values_dp1_, 1)  # will be (B, K, 1)
            accum_values = (coord_values_d + coord_values_dp1) / 2
            max_coord_values_dp1, max_coords_dp1 = torch.max(accum_values, 1, keepdim=True)  # (B,1)
            coords_dp1_ = coord_idxs_d.gather(1, max_coords_dp1.repeat(1, 1, 3))
            return coords_dp1_, max_coord_values_dp1, voxel_grid, q, q_rot_grip, rot_grip_index
        else:
            return coord_idxs_d, coord_values_d, voxel_grid, q, q_rot_grip, rot_grip_index


    def _extract_crop(self, pixel_action, observation):
        b, t, c, h, w = observation.shape
        observation = observation.view(b * t, c, h, w)
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0,
            h - self._image_crop_size)
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode='nearest',
                             align_corners=True)
        return crop.view((b, t) + crop.shape[1:])

    def _preprocess_inputs(self, replay_sample):
        obs, obs_tp1 = [], []
        pcds, pcds_tp1 = [], []
        self._crop_summary, self._crop_summary_tp1 = [], []
        for n in self._camera_names:
            if self._layer > 0 and 'wrist' not in n:
                pc_t = replay_sample['%s_pixel_coord' % n].view(-1, 1, 2)
                pc_tp1 = replay_sample['%s_pixel_coord_tp1' % n].view(-1, 1, 2)
                rgb = self._extract_crop(pc_t, replay_sample['%s_rgb' % n])
                rgb_tp1 = self._extract_crop(pc_tp1,
                                             replay_sample['%s_rgb_tp1' % n])
                pcd = self._extract_crop(pc_t,
                                         replay_sample['%s_point_cloud' % n])
                pcd_tp1 = self._extract_crop(pc_tp1, replay_sample[
                    '%s_point_cloud_tp1' % n])
                self._crop_summary.append((n, rgb[:, -1]))
                self._crop_summary_tp1.append(('%s_tp1' % n, rgb_tp1[:, -1]))
            else:
                rgb = (replay_sample['%s_rgb' % n])
                rgb_tp1 = (replay_sample['%s_rgb_tp1' % n])
                pcd = (replay_sample['%s_point_cloud' % n])
                pcd_tp1 = (replay_sample['%s_point_cloud_tp1' % n])
            obs.append([rgb, pcd])
            obs_tp1.append([rgb_tp1, pcd_tp1])
            pcds.append(pcd)
            pcds_tp1.append(pcd_tp1)
        return obs, obs_tp1, pcds, pcds_tp1

    def _act_preprocess_inputs(self, observation):
        obs, pcds = [], []
        for n in self._camera_names:
            rgb = (observation['%s_rgb' % n])
            pcd = (observation['%s_point_cloud' % n])
            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].long()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        q_rot = torch.stack(torch.split(
            rot_grip_q[:, :-2], int(360 // self._rotation_resolution),
            dim=1), dim=1)  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
             q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
             q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
             q_grip.gather(1, rot_and_grip_idx[:, 3:4])], -1)
        return rot_and_grip_values

    def update(self, step: int, replay_sample: dict) -> dict:

        action_trans = replay_sample['trans_action_indicies'][:, -1,
                       self._layer * 3:self._layer * 3 + 3]
        action_rot_grip = replay_sample['rot_grip_action_indicies']
        b, t = action_rot_grip.shape[:2]
        action_rot_grip = action_rot_grip[:, -1].long()
        reward = replay_sample['reward'] * 0.01
        reward = torch.where(reward >= 0, reward, torch.zeros_like(reward))

        if self._layer > 0:
            cp = replay_sample['attention_coordinate_layer_%d' % (self._layer - 1)].view(b * t, -1)
            cp_tp1 = replay_sample['attention_coordinate_layer_%d_tp1' % (self._layer - 1)].view(b * t, -1)
            bounds = torch.cat(
                [cp - self._bounds_offset, cp + self._bounds_offset], dim=1)
            bounds_tp1 = torch.cat(
                [cp_tp1 - self._bounds_offset, cp_tp1 + self._bounds_offset],
                dim=1)
        else:
            bounds = bounds_tp1 = self._coordinate_bounds.repeat(b * t, 1)

        proprio = proprio_tp1 = None
        if self._include_low_dim_state:
            proprio = stack_on_channel(replay_sample['low_dim_state'])
            proprio_tp1 = stack_on_channel(replay_sample['low_dim_state_tp1'])

        terminal = replay_sample['terminal'].float()

        obs, obs_tp1, pcd, pcd_tp1 = self._preprocess_inputs(replay_sample)

        q, q_rot_grip, voxel_grid = self._q(
            obs, proprio, pcd, bounds,
            replay_sample.get('prev_layer_voxel_grid', None))
        coords, rot_and_grip_indicies = self._q.choose_highest_action(q, q_rot_grip)

        with_rot_and_grip = rot_and_grip_indicies is not None

        with torch.no_grad():

            if not self._tree_during_update:
                q_tp1_targ, q_rot_grip_tp1_targ, _ = self._q_target(
                    obs_tp1, proprio_tp1, pcd_tp1, bounds_tp1,
                    replay_sample.get('prev_layer_voxel_grid_tp1', None))
                q_tp1, q_rot_grip_tp1, voxel_grid_tp1 = self._q(
                    obs_tp1, proprio_tp1, pcd_tp1, bounds_tp1,
                    replay_sample.get('prev_layer_voxel_grid_tp1', None))
                coords_tp1, rot_and_grip_indicies_tp1 = self._q.choose_highest_action(q_tp1, q_rot_grip_tp1)
                q_tp1_at_voxel_idx = self._get_value_from_voxel_index(q_tp1_targ, coords_tp1)
            else:
                coord_idxs, q_tp1_at_voxel_idx, voxel_grid_tp1, _, q_rot_grip_tp1_targ, rot_and_grip_indicies_tp1 = self.get_highest_tree_value(
                obs_tp1, proprio_tp1, pcd_tp1, cp_tp1 if self._layer > 0 else None, index_target=True, prev_vox=replay_sample.get('prev_layer_voxel_grid_tp1', None))
                q_tp1_at_voxel_idx = q_tp1_at_voxel_idx[:, 0]

            if with_rot_and_grip:
                q_tp1_at_voxel_idx = torch.cat([q_tp1_at_voxel_idx, q_rot_grip_tp1_targ], 1).mean(1, keepdim=True)

            q_target = (reward.unsqueeze(1) + (self._gamma ** self._nstep) * (1 - terminal.unsqueeze(1)) * q_tp1_at_voxel_idx).detach()
            q_target = torch.maximum(q_target, torch.zeros_like(q_target))

        qreg_loss = F.l1_loss(q, torch.zeros_like(q), reduction='none')
        qreg_loss = qreg_loss.mean(-1).mean(-1).mean(-1).mean(-1) * self._lambda_trans_qreg
        chosen_trans_q1 = self._get_value_from_voxel_index(q, action_trans)
        q_delta = F.smooth_l1_loss(chosen_trans_q1[:, :1], q_target[:, :1], reduction='none')
        if with_rot_and_grip:
            target_q_rot_grip = self._get_value_from_rot_and_grip(q_rot_grip, action_rot_grip)  # (B, 4)
            q_delta = torch.cat([F.smooth_l1_loss(target_q_rot_grip, q_target.repeat((1, 4)), reduction='none'), q_delta], -1)
            qreg_loss += F.l1_loss(q_rot_grip, torch.zeros_like(q_rot_grip), reduction='none').mean(1) * self._lambda_rot_qreg

        loss_weights = utils.loss_weights(replay_sample, REPLAY_BETA)
        combined_delta = q_delta.mean(1)
        total_loss = ((combined_delta + qreg_loss) * loss_weights).mean()

        self._optimizer.zero_grad()
        total_loss.backward()
        if self._grad_clip is not None:
            nn.utils.clip_grad_value_(self._q.parameters(), self._grad_clip)
        self._optimizer.step()

        self._summaries = {
            'q/mean_qattention': q.mean(),
            'q/max_qattention': chosen_trans_q1.max(1)[0].mean(),
            'losses/total_loss': total_loss,
            'losses/qreg': qreg_loss.mean()
        }
        if with_rot_and_grip:
            self._summaries.update({
                'q/mean_q_rotation': q_rot_grip.mean(),
                'q/max_q_rotation': target_q_rot_grip[:, :3].mean(),
                'losses/bellman_rotation': q_delta[:, 3].mean(),
                'losses/bellman_gripper': q_delta[:, 3].mean(),
                'losses/bellman_qattention': q_delta[:, 4].mean(),
            })
        else:
            self._summaries.update({
                'losses/bellman_qattention': q_delta.mean(),
            })

        self._vis_voxel_grid = voxel_grid[0, -1]
        self._vis_translation_qvalue = q[0]
        self._vis_max_coordinate = coords[0]

        utils.soft_updates(self._q, self._q_target, self._tau)
        priority = (combined_delta + 1e-10).sqrt()
        priority /= priority.max()
        prev_priority = replay_sample.get('priority', 0)

        return {
            'priority': priority + prev_priority,
            'prev_layer_voxel_grid': voxel_grid,
            'prev_layer_voxel_grid_tp1': voxel_grid_tp1,
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        deterministic = True  # TODO: Don't explicitly explore.
        bounds = self._coordinate_bounds.repeat(self._timesteps, 1)

        if self._layer > 0:
            cp = observation['attention_coordinate']
            bounds = torch.cat(
                [cp - self._bounds_offset, cp + self._bounds_offset], dim=1)
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size

        max_rot_index = int(360 // self._rotation_resolution)
        proprio = None
        if self._include_low_dim_state:
            proprio = stack_on_channel(observation['low_dim_state'])
        obs, pcd = self._act_preprocess_inputs(observation)

        if self._tree_during_act:
            coord_idxs, _, vox_grid, q, q_rot_grip, rot_and_grip_indicies = self.get_highest_tree_value(
                obs, proprio, pcd, cp if self._layer > 0 else None, observation.get('prev_layer_voxel_grid', None))
            coord_idxs = coord_idxs.int()[:, 0]  # k axis will be 1 here
        else:
            q, q_rot_grip, vox_grid = self._q(obs, proprio, pcd, bounds, observation.get( 'prev_layer_voxel_grid', None))
            coords, rot_and_grip_indicies = self._q.choose_highest_action(q, q_rot_grip)
            coord_idxs = coords.int()

        rot_grip_action = rot_and_grip_indicies
        attention_coordinate = bounds[:, :3] + res * coord_idxs + res / 2

        observation_elements = {
            'attention_coordinate': attention_coordinate,
            'prev_layer_voxel_grid': vox_grid,
        }
        info = {
            'voxel_grid_depth%d' % self._layer: vox_grid,
            'q_depth%d' % self._layer: q,
            'voxel_idx_depth%d' % self._layer: coord_idxs
        }
        self._act_voxel_grid = vox_grid[0, -1]
        self._act_max_coordinate = coord_idxs[0]
        self._act_qvalues = q[0]
        return ActResult((coord_idxs, rot_grip_action),
                         observation_elements=observation_elements,
                         info=info)

    def update_summaries(self) -> List[Summary]:
        summaries = [
            ImageSummary('%s/update_qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._vis_voxel_grid.detach().cpu().numpy(),
                             self._vis_translation_qvalue.detach().cpu().numpy(),
                             self._vis_max_coordinate.detach().cpu().numpy())))
        ]

        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))

        for (name, crop) in (self._crop_summary + self._crop_summary_tp1):
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.extend([
                ImageSummary('%s/crops/%s' % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            assert not torch.isnan(param.grad.abs() <= 1.0).all()
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (self._name, tag),
                                 param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (self._name, tag),
                                 param.data))

        for name, t in self._q.latents().items():
            summaries.append(
                HistogramSummary('%s/activations/%s' % (self._name, name), t))

        return summaries

    def act_summaries(self) -> List[Summary]:
        return [
            ImageSummary('%s/act_Qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._act_voxel_grid.cpu().numpy(),
                             self._act_qvalues.cpu().numpy(),
                             self._act_max_coordinate.cpu().numpy())))]

    def load_weights(self, savedir: str):
        self._q.load_state_dict(
            torch.load(os.path.join(savedir, '%s.pt' % self._name),
                       map_location=self._device))

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % self._name))
