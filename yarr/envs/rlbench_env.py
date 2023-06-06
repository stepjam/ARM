from typing import Type, List

import numpy as np
try:
    from rlbench import ObservationConfig, Environment, CameraConfig
except (ModuleNotFoundError, ImportError) as e:
    print("You need to install RLBench: 'https://github.com/stepjam/RLBench'")
    raise e
from rlbench.action_modes import ActionMode
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task

from yarr.envs.env import Env
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition


class RLBenchEnv(Env):

    ROBOT_STATE_KEYS = [
        'joint_velocities', 'joint_positions', 'joint_forces', 'gripper_open',
        'gripper_pose', 'gripper_joint_positions', 'gripper_touch_forces',
        'task_low_dim_state', 'misc'
    ]

    def __init__(self,
                 task_class: Type[Task],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True):
        super(RLBenchEnv, self).__init__()
        self._task_class = task_class
        self._observation_config = observation_config
        self._channels_last = channels_last
        self._rlbench_env = Environment(action_mode=action_mode,
                                        obs_config=observation_config,
                                        dataset_root=dataset_root,
                                        headless=headless)
        self._task = None

    def extract_obs(self, obs: Observation):
        obs_dict = vars(obs)
        obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
        robot_state = obs.get_low_dim_data()
        # Remove all of the individual state elements
        obs_dict = {
            k: v
            for k, v in obs_dict.items()
            if k not in RLBenchEnv.ROBOT_STATE_KEYS
        }
        if not self._channels_last:
            # Swap channels from last dim to 1st dim
            obs_dict = {
                k: np.transpose(v, [2, 0, 1])
                if v.ndim == 3 else np.expand_dims(v, 0)
                for k, v in obs_dict.items()
            }
        else:
            # Add extra dim to depth data
            obs_dict = {
                k: v if v.ndim == 3 else np.expand_dims(v, -1)
                for k, v in obs_dict.items()
            }
        obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)
        return obs_dict

    def launch(self):
        self._rlbench_env.launch()
        self._task = self._rlbench_env.get_task(self._task_class)

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self) -> dict:
        descriptions, obs = self._task.reset()
        return self.extract_obs(obs)

    def step(self, action: np.ndarray) -> Transition:
        obs, reward, terminal = self._task.step(action)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    def _get_cam_observation_elements(self, camera: CameraConfig, prefix: str):
        elements = []
        if camera.rgb:
            shape = (camera.image_size + (3, ) if self._channels_last else
                     (3, ) + camera.image_size)
            elements.append(
                ObservationElement('%s_rgb' % prefix, shape, np.uint8))
        if camera.depth:
            shape = (camera.image_size + (1, ) if self._channels_last else
                     (1, ) + camera.image_size)
            elements.append(
                ObservationElement('%s_depth' % prefix, shape, np.float32))
        if camera.mask:
            raise NotImplementedError()
        return elements

    @property
    def observation_elements(self) -> List[ObservationElement]:
        elements = []
        robot_state_len = 0
        if self._observation_config.joint_velocities:
            robot_state_len += 7
        if self._observation_config.joint_positions:
            robot_state_len += 7
        if self._observation_config.joint_forces:
            robot_state_len += 7
        if self._observation_config.gripper_open:
            robot_state_len += 1
        if self._observation_config.gripper_pose:
            robot_state_len += 7
        if self._observation_config.gripper_joint_positions:
            robot_state_len += 2
        if self._observation_config.gripper_touch_forces:
            robot_state_len += 2
        if self._observation_config.task_low_dim_state:
            raise NotImplementedError()
        if robot_state_len > 0:
            elements.append(
                ObservationElement('low_dim_state', (robot_state_len, ),
                                   np.float32))
        elements.extend(
            self._get_cam_observation_elements(
                self._observation_config.left_shoulder_camera,
                'left_shoulder'))
        elements.extend(
            self._get_cam_observation_elements(
                self._observation_config.right_shoulder_camera,
                'right_shoulder'))
        elements.extend(
            self._get_cam_observation_elements(
                self._observation_config.front_camera, 'front'))
        elements.extend(
            self._get_cam_observation_elements(
                self._observation_config.wrist_camera, 'wrist'))
        return elements

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment:
        return self._rlbench_env
