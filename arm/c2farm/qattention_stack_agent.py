
from typing import List

import torch
from yarr.agents.agent import Agent, ActResult, Summary

import numpy as np

from arm import utils
from arm.c2farm.qattention_agent import QAttentionAgent

NAME = 'QAttentionStackAgent'
GAMMA = 0.99
NSTEP = 1
REPLAY_ALPHA = 0.7
REPLAY_BETA = 0.5


class QAttentionStackAgent(Agent):

    def __init__(self,
                 qattention_agents: List[QAttentionAgent],
                 rotation_resolution: float,
                 camera_names: List[str],
                 rotation_prediction_depth: int = 0):
        super(QAttentionStackAgent, self).__init__()
        self._qattention_agents = qattention_agents
        self._rotation_resolution = rotation_resolution
        self._camera_names = camera_names
        self._rotation_prediction_depth = rotation_prediction_depth

    def build(self, training: bool, device=None) -> None:
        '''
        iteratively build the network for each agent

        Input:
        - training: whether the agent is in the training mode
        - device: 
        '''

        self._device = device
        if self._device is None:
            self._device = torch.device('cpu')
        for qa in self._qattention_agents:
            qa.build(training, device)

    def update(self, step: int, replay_sample: dict) -> dict:
        priorities = 0
        for qa in self._qattention_agents:
            update_dict = qa.update(step, replay_sample)
            priorities += update_dict['priority']
            replay_sample.update(update_dict)
        return {
            'priority': (priorities) ** REPLAY_ALPHA,
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:

        observation_elements = {}
        translation_results, rot_grip_results = [], []
        infos = {}
        for depth, qagent in enumerate(self._qattention_agents):
            act_results = qagent.act(step, observation, deterministic)
            attention_coordinate = act_results.observation_elements['attention_coordinate'].cpu().numpy()
            observation_elements['attention_coordinate_layer_%d' % depth] = attention_coordinate[0]

            translation_idxs, rot_grip_idxs = act_results.action
            translation_results.append(translation_idxs)
            if rot_grip_idxs is not None:
                rot_grip_results.append(rot_grip_idxs)

            observation['attention_coordinate'] = act_results.observation_elements['attention_coordinate']
            observation['prev_layer_voxel_grid'] = act_results.observation_elements['prev_layer_voxel_grid']

            for n in self._camera_names:
                px, py = utils.point_to_pixel_index(
                    attention_coordinate[0],
                    observation['%s_camera_extrinsics' % n][0, 0].cpu().numpy(),
                    observation['%s_camera_intrinsics' % n][0, 0].cpu().numpy())
                pc_t = torch.tensor([[[py, px]]], dtype=torch.float32, device=self._device)
                observation['%s_pixel_coord' % n] = pc_t
                observation_elements['%s_pixel_coord' % n] = [py, px]

            infos.update(act_results.info)

        rgai = torch.cat(rot_grip_results, 1)[0].cpu().numpy()
        observation_elements['trans_action_indicies'] = torch.cat(translation_results, 1)[0].cpu().numpy()
        observation_elements['rot_grip_action_indicies'] = rgai
        continuous_action = np.concatenate([
            act_results.observation_elements['attention_coordinate'].cpu().numpy()[0],
            utils.discrete_euler_to_quaternion(rgai[-4:-1], self._rotation_resolution),
            rgai[-1:]])
        return ActResult(
            continuous_action,
            observation_elements=observation_elements,
            info=infos
        )

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for qa in self._qattention_agents:
            summaries.extend(qa.update_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        s = []
        for qa in self._qattention_agents:
            s.extend(qa.act_summaries())
        return s

    def load_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.load_weights(savedir)

    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)
