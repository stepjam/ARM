from typing import List

import numpy as np
import torch
from yarr.agents.agent import Agent, ActResult, Summary, ScalarSummary

from arm import utils
from arm.custom_rlbench_env import CustomRLBenchEnv
from arm.lpr.const import UNKNOWN_COLLISION
from arm.lpr.rlbench_path_sampler import RLBenchPathSampler

IK_STEPS_ON_PATH = 50


class CombinedAgent(Agent):

    def __init__(self,
                 trajectory_agent,
                 stack_agent,
                 env: CustomRLBenchEnv,
                 trajectory_point_noise: float,
                 trajectory_points: int,
                 trajectory_mode: str,
                 trajectory_samples: int,
                 learn_trajectory_pi: bool):
        super(CombinedAgent, self).__init__()
        self._trajectory_agent = trajectory_agent
        self._stack_agent = stack_agent
        self._env = env
        self._trajectory_point_noise = trajectory_point_noise
        self._trajectory_points = trajectory_points
        self._trajectory_mode = trajectory_mode
        self._trajectory_samples = trajectory_samples
        self._learn_trajectory_pi = learn_trajectory_pi
        self._sampler = RLBenchPathSampler(
            trajectory_points, trajectory_samples, trajectory_point_noise)
        self._num_valid_chosen = self._num_valid = self._num_steps = 0

    def build(self, training: bool, device=None) -> None:
        self._device = device
        if self._device is None:
            self._device = torch.device('cpu')
        for qa in [self._trajectory_agent, self._stack_agent]:
            qa.build(training, device)

    def update(self, step: int, replay_sample: dict) -> dict:
        priorities = 0
        for qa in [self._trajectory_agent, self._stack_agent]:
            update_dict = qa.update(step, replay_sample)
            priorities += update_dict['priority']
            replay_sample.update(update_dict)
        return {
            'priority': priorities,
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        next_best_pose_result = self._stack_agent.act(step, observation, deterministic)
        nbp = next_best_pose_result.action
        grip = nbp[-1:]
        valid_cart_paths, valid_config_paths = self._sampler.sample_paths_to_valued(nbp[:-1])

        st = np.linspace(observation['gripper_pose'].detach().cpu().numpy()[0, 0],
                         nbp[:-1], self._trajectory_points)
        st[:, 3:] = utils.normalize_quaternion(st[:, 3:])
        observation['linspace'] = torch.from_numpy(st).to(self._device).unsqueeze(0)

        valid_pi = False
        chosen_cfg_traj = np.zeros((self._trajectory_points * 7 + 3,))
        chosen_traj = np.zeros((self._trajectory_points * 7 + 3,))
        if self._learn_trajectory_pi:
            observation['nbp'] = torch.from_numpy(np.array(nbp[:-1], dtype=np.float32)).to(self._device).unsqueeze(0)
            act_results = self._trajectory_agent.act_pi(step, observation, deterministic)
            cfg_path = act_results.action.cpu()[0].view(-1, 7).numpy()
            path_length = np.abs(cfg_path[:-1] - cfg_path[1:]).sum()
            if len(valid_config_paths) > 0:
                vcp = np.array(valid_config_paths)
                mean_path_len = np.abs(vcp[:, :-1] - vcp[:, 1:]).sum(-1).sum(-1).mean()
                if path_length <= mean_path_len:
                    chosen_cfg_traj = np.concatenate([cfg_path.reshape((-1,)), UNKNOWN_COLLISION])
                    cart = conf = chosen_cfg_traj
                    if cart is not None:
                        valid_cart_paths.insert(0, cart)
                        valid_config_paths.insert(0, conf)
                        self._num_valid += 1
                        valid_pi = True

        if len(valid_cart_paths) > 0:
            qin = valid_cart_paths if self._trajectory_mode == 'pose' else valid_config_paths
            observation['trajectory_t'] = torch.from_numpy(np.array(qin, dtype=np.float32)).to(self._device).unsqueeze(1)
            traj_result = self._trajectory_agent.act(step, observation, deterministic)
            values = traj_result.action.cpu().numpy()[:, 0]
            traj_argmax = np.argmax(values)
            chosen_traj = valid_cart_paths[traj_argmax]
            chosen_cfg_traj = valid_config_paths[traj_argmax]

            if valid_pi and traj_argmax == 0:
                self._num_valid_chosen += 1

        self._num_steps += 1
        next_best_pose_result.observation_elements['trajectory'] = (
            chosen_traj if self._trajectory_mode == 'pose' else chosen_cfg_traj)
        next_best_pose_result.action = np.concatenate([chosen_cfg_traj[:-3], grip])
        next_best_pose_result.replay_elements['linspace'] = st
        return next_best_pose_result

    def update_summaries(self) -> List[Summary]:
        summaries = []
        for qa in [self._trajectory_agent, self._stack_agent]:
            summaries.extend(qa.update_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        s = []
        if self._learn_trajectory_pi and self._num_steps > 0:
            s.append(ScalarSummary('combined_agent/pi_valid',
                                   float(self._num_valid / self._num_steps)))
            s.append(ScalarSummary('combined_agent/pi_valid_chosen',
                                   float(self._num_valid_chosen / self._num_steps)))
            self._num_valid_chosen = self._num_valid = self._num_steps = 0
        for qa in [self._trajectory_agent, self._stack_agent]:
            s.extend(qa.act_summaries())
        return s

    def load_weights(self, savedir: str):
        for qa in [self._trajectory_agent, self._stack_agent]:
            qa.load_weights(savedir)

    def save_weights(self, savedir: str):
        for qa in [self._trajectory_agent, self._stack_agent]:
            qa.save_weights(savedir)
