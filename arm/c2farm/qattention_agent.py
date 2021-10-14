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
        '''
        calculate the index of x-axis, y-axis, z-axis with the maximum 
        q value of the voxel grid
        
        Input:
        - tensor_orig: the q value voxel grid with shape (batch_size, 
                       channel_size, voxel_size, voxel_size, voxel_size)
        
        Output:
        - indices: index of the voxel with highest q-value in the voxel grid 
                   with shape (batch_size, 3)
        '''

        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip):
        '''
        choose the voxel with the highest q value\\
        If `q_rot_grip` is `None`, `rot_and_grip_indicies` will be `None` as well
        
        Input:
        - q_trans: the q value voxel grid with shape (batch_size, 
                   channel_size, voxel_size, voxel_size, voxel_size)
        - q_rot_grip: (batch_size, 360//rotation_resolution*3)
        
        Output:
        - coords: position (index) voxel field with highest q-value with 
                  shape (batch_size, 3)
        - rot_and_grip_indicies: rotation index for eular (xyz) and 
                                 whether the gripper is open (0 or 1) with shape  
                                 (batch_size, 4)
        '''

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

    def forward(self, x, proprio, pcd,
                bounds=None, latent=None):
        '''
        Input:
        - x (list): [rgb, pcd]
        - proprio: the state for the robot arm
        - pcd: point cloud
        - bound
        - latent    
        
        Output:
        - q_trans: the q value voxel grid with shape (batch_size, 
                   channel_size, voxel_size, voxel_size, voxel_size)
        - rot_and_grip_q: (batch_size, 360//rotation_resolution*3)
        - voxel_grid: the voxel grid formed by observation (rgb and 
                      depth image) with shape (batch_size, voxel_feat, 
                      voxel_size, voxel_size, voxel_size)
        '''

        # x will be list of list (list of [rgb, pcd])
        b = x[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        image_features = [xx[0] for xx in x]
        feat_size = image_features[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in
             image_features], 1)

        voxel_grid = self._voxel_grid.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)

        # Swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

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
                 lambda_weight_l2: float = 0.0
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
        self._exploration_strategy = exploration_strategy
        self._lambda_weight_l2 = lambda_weight_l2

        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution

        self._name = NAME + '_layer' + str(self._layer)

    def build(self, training: bool, device: torch.device = None):
        '''
        build the network

        Input:
        - training: whether the built network is in the training mode
        - device: the device (gpu/cpu) used for the agent
        '''

        if device is None:
            device = torch.device('cpu')

        vox_grid = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
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

    def _extract_crop(self, pixel_action, observation):
        '''
        use the `pixel_action` as anchor

        Input:
        - pixel_action: (batch_size, 1, 2)
        - observation: (batch_size, 1, 3, img_h, img_w)

        Output:
        - crop:
        '''

        # Pixel action will now be (B, 2)
        observation = stack_on_channel(observation)
        h = observation.shape[-1]
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
        return crop

    def _preprocess_inputs(self, replay_sample):
        '''
        pack the inputs

        If the layer > 0, we will crop the rgb/depth image

        Input:
        - replay_sample: the sampled transitions from the replay buffer

        Output:
        - obs: rgb and depth image 
        - obs_tp1: rgb and depth image for next timestep
        - pcds: depth image 
        - pcds_tp1: depth image for next timestep
        '''


        obs, obs_tp1 = [], []
        pcds, pcds_tp1 = [], []
        self._crop_summary, self._crop_summary_tp1 = [], []
        for n in self._camera_names:
            if self._layer > 0 and 'wrist' not in n:
                pc_t = replay_sample['%s_pixel_coord' % n]
                pc_tp1 = replay_sample['%s_pixel_coord_tp1' % n]
                rgb = self._extract_crop(pc_t, replay_sample['%s_rgb' % n])
                rgb_tp1 = self._extract_crop(pc_tp1,
                                             replay_sample['%s_rgb_tp1' % n])
                pcd = self._extract_crop(pc_t,
                                         replay_sample['%s_point_cloud' % n])
                pcd_tp1 = self._extract_crop(pc_tp1, replay_sample[
                    '%s_point_cloud_tp1' % n])
                self._crop_summary.append((n, rgb))
                self._crop_summary_tp1.append(('%s_tp1' % n, rgb_tp1))
            else:
                rgb = stack_on_channel(replay_sample['%s_rgb' % n])
                rgb_tp1 = stack_on_channel(replay_sample['%s_rgb_tp1' % n])
                pcd = stack_on_channel(replay_sample['%s_point_cloud' % n])
                pcd_tp1 = stack_on_channel(
                    replay_sample['%s_point_cloud_tp1' % n])
            obs.append([rgb, pcd])
            obs_tp1.append([rgb_tp1, pcd_tp1])
            pcds.append(pcd)
            pcds_tp1.append(pcd_tp1)
        return obs, obs_tp1, pcds, pcds_tp1

    def _act_preprocess_inputs(self, observation):
        '''
        pack the observation for each camera

        Input:
        - obervation:
            - front_rgb:
            - front_point_cloud: 
            - low_dim_state:
            - front_camera_extrinsics:
            - front_camera_intrinsics:

            If the depth > 0

            - attention_coordinate:
            - prev_layer_voxel_grid:
            - front_pixel_coord:

        Output:
        - obs (list): [[rgb, pcd], [rgb, pcd], ......]
        - pcds: list of point cloud (with shape (batch_size, 3, h, w))
        '''

        obs, pcds = [], []
        for n in self._camera_names:
            if self._layer > 0 and 'wrist' not in n:
                pc_t = observation['%s_pixel_coord' % n]
                rgb = self._extract_crop(pc_t, observation['%s_rgb' % n])
                pcd = self._extract_crop(pc_t, observation['%s_point_cloud' % n])
            else:
                rgb = stack_on_channel(observation['%s_rgb' % n])
                pcd = stack_on_channel(observation['%s_point_cloud' % n])
            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        '''
        extract the feature from the voxel grid feature with index
        
        Input:
        - q: a voxel-grid of the q value (batch_size, channel_size, 
             voxel_size, voxel_size, voxel_size)
        - voxel_idx: (batch_size, 3)

        Output:
        - chosen_voxel_values: (batch_size, channel_size)
        '''


        b, c, d, h, w = q.shape
        q_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].long()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        '''

        
        Input:
        - rot_grip_q: (batch_size, 360//rotation_resolution*3 + 2)
        - rot_and_grip_idx: (batch_size, 4)

        Output:
        - rot_and_grip_values: (batch_size, 4)
        '''

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
        '''
        update the policy parameters

        NOTE: 'tp1' means next state

        Input:
        - step:
        - replay_sample (dict): contains the sampled transitions
        '''


        action_trans = replay_sample['trans_action_indicies'][:, -1,
                       self._layer * 3:self._layer * 3 + 3]
        action_rot_grip = replay_sample['rot_grip_action_indicies'][:, -1].long()
        reward = replay_sample['reward'] * 0.01
        reward = torch.where(reward >= 0, reward, torch.zeros_like(reward))

        bounds = bounds_tp1 = self._coordinate_bounds
        if self._layer > 0:
            cp = replay_sample['attention_coordinate_layer_%d' % (
                    self._layer - 1)][:, -1]
            cp_tp1 = replay_sample[
                         'attention_coordinate_layer_%d_tp1' % (
                                 self._layer - 1)][:, -1]
            bounds = torch.cat(
                [cp - self._bounds_offset, cp + self._bounds_offset], dim=1)
            bounds_tp1 = torch.cat(
                [cp_tp1 - self._bounds_offset, cp_tp1 + self._bounds_offset],
                dim=1)

        proprio = proprio_tp1 = None
        if self._include_low_dim_state:
            proprio = stack_on_channel(replay_sample['low_dim_state'])
            proprio_tp1 = stack_on_channel(replay_sample['low_dim_state_tp1'])

        # Don't want timeouts to be classed as terminals
        terminal = replay_sample['terminal'].float() - replay_sample['timeout'].float()

        obs, obs_tp1, pcd, pcd_tp1 = self._preprocess_inputs(replay_sample)

        q, q_rot_grip, voxel_grid = self._q(
            obs, proprio, pcd, bounds,
            replay_sample.get('prev_layer_voxel_grid', None))
        coords, rot_and_grip_indicies = self._q.choose_highest_action(q, q_rot_grip)

        with_rot_and_grip = rot_and_grip_indicies is not None

        with torch.no_grad():
            q_tp1_targ, q_rot_grip_tp1_targ, _ = self._q_target(
                obs_tp1, proprio_tp1, pcd_tp1, bounds_tp1,
                replay_sample.get('prev_layer_voxel_grid_tp1', None))

            q_tp1, q_rot_grip_tp1, voxel_grid_tp1 = self._q(
                obs_tp1, proprio_tp1, pcd_tp1, bounds_tp1,
                replay_sample.get('prev_layer_voxel_grid_tp1', None))
            coords_tp1, rot_and_grip_indicies_tp1 = self._q.choose_highest_action(q_tp1, q_rot_grip_tp1)

            q_tp1_at_voxel_idx = self._get_value_from_voxel_index(q_tp1_targ, coords_tp1)
            if with_rot_and_grip:
                target_q_tp1_rot_grip = self._get_value_from_rot_and_grip(q_rot_grip_tp1_targ, rot_and_grip_indicies_tp1)  # (B, 4)
                q_tp1_at_voxel_idx = target_q_tp1_rot_grip.mean(1, keepdim=True)

            q_target = (reward.unsqueeze(1) + (self._gamma ** self._nstep) * (1 - terminal.unsqueeze(1)) * q_tp1_at_voxel_idx).detach()
            q_target = torch.clamp(q_target, 0.0, 1.0)

        qreg_loss = F.l1_loss(q, torch.zeros_like(q), reduction='none')
        qreg_loss = qreg_loss.mean(-1).mean(-1).mean(-1).mean(-1) * self._lambda_trans_qreg
        chosen_trans_q1 = self._get_value_from_voxel_index(q, action_trans)
        q_delta = F.smooth_l1_loss(chosen_trans_q1[:, :1], q_target, reduction='none')
        if with_rot_and_grip:
            target_q_rot_grip = self._get_value_from_rot_and_grip(q_rot_grip, action_rot_grip)  # (B, 4)
            q_delta = torch.cat([F.smooth_l1_loss(target_q_rot_grip, q_target.repeat((1, 4)), reduction='none'), q_delta], -1)
            qreg_loss += F.l1_loss(q_rot_grip, torch.zeros_like(q_rot_grip), reduction='none').mean(1) * self._lambda_trans_qreg

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

        self._vis_voxel_grid = voxel_grid[0]
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
        '''
        take the observation as input and reture the action

        Input:
        - step: dummy, please neglect
        - observation (dict):
            - front_rgb: rgb image of front camera
            - front_point_cloud: depth image of front camera
            - low_dim_state: the state of the robot arm 
            - front_camera_extrinsics
            - front_camera_intrinsics 

            (when layer>=1, the following information is included)
            - attention_coordinate
            - prev_layer_voxel_grid
            - front_pixel_coord

        Output:
        - act_result:
            - action (tuple): it contain `coords` and `rot_grip_action`
                - coords: the gripper position in the voxel grid index
                - rot_grip_action:
            - observation_elements (dict):
                - attention_coordinate:
                - prev_layer_voxel_grid:
            - info (dict):
                - voxel_grid_depth:
                - q_depth:
                - voxel_idx_depth:
        '''


        bounds = self._coordinate_bounds

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

        # coords: (1, 3)
        q, q_rot_grip, vox_grid = self._q(obs, proprio, pcd, bounds,
                              observation.get('prev_layer_voxel_grid', None))
        coords, rot_and_grip_indicies = self._q.choose_highest_action(q, q_rot_grip)


        rot_grip_action = rot_and_grip_indicies

        if (not deterministic) and self._exploration_strategy == 'gaussian':
            trans_noise = torch.round(torch.normal(0.0, 1, size=(1, 3)))
            coords = torch.clamp(coords + trans_noise, 0,
                                 self._voxel_size - 1)

            rg_noise = torch.round(torch.normal(0.0, 1, size=(1, 3)))
            if rot_grip_action is not None:
                explore_rot = torch.clamp(
                    rot_and_grip_indicies[:, :3] + rg_noise,
                    0, max_rot_index - 1)
                grip = rot_and_grip_indicies[:, 3:]
                # For now, randomly swap gripper 20% of time
                if np.random.random() < 0.2:
                    grip = torch.randint(0, 2, size=(1, 1))
                # For now, randomly swap gripper 20% of time
                rot_grip_action = torch.cat([explore_rot, grip], -1)

        coords = coords.int()
        attention_coordinate = bounds[:, :3] + res * coords + res / 2
        observation_elements = {
            'attention_coordinate': attention_coordinate,
            'prev_layer_voxel_grid': vox_grid,
        }
        info = {
            'voxel_grid_depth%d' % self._layer: vox_grid,
            'q_depth%d' % self._layer: q,
            'voxel_idx_depth%d' % self._layer: coords
        }
        self._act_voxel_grid = vox_grid[0]
        self._act_max_coordinate = coords[0]
        self._act_qvalues = q[0]
        return ActResult((coords, rot_grip_action),
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
