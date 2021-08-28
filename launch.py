import os
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from typing import List
import torch
from pyrep.const import RenderMode
from rlbench import CameraConfig, ObservationConfig, ArmActionMode
from rlbench.action_modes import ActionMode, GripperActionMode
from rlbench.backend import task
from rlbench.backend.utils import task_file_to_task_class
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners.pytorch_train_runner import PyTorchTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from arm import arm
from arm import c2farm
from arm.baselines import bc, td3, dac, sac
from arm.custom_rlbench_env import CustomRLBenchEnv
import numpy as np

import hydra
import logging
from omegaconf import DictConfig, OmegaConf, ListConfig


def _create_obs_config(camera_names: List[str], camera_resolution: List[int]):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=False,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config


def _modify_action_min_max(action_min_max):
    # Make translation bounds a little bigger
    action_min_max[0][0:3] -= np.fabs(action_min_max[0][0:3]) * 0.2
    action_min_max[1][0:3] += np.fabs(action_min_max[1][0:3]) * 0.2
    action_min_max[0][-1] = 0
    action_min_max[1][-1] = 1
    action_min_max[0][3:7] = np.array([-1, -1, -1, 0])
    action_min_max[1][3:7] = np.array([1, 1, 1, 1])
    return action_min_max


def _get_device(gpu):
    if gpu is not None and gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:%d" % gpu)
        torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device


def run_seed(cfg: DictConfig, env, cams, train_device, env_device, seed) -> None:
    train_envs = cfg.framework.train_envs
    replay_ratio = None if cfg.framework.replay_ratio == 'None' else cfg.framework.replay_ratio
    replay_split = [1]
    replay_path = os.path.join(cfg.replay.path, cfg.rlbench.task, cfg.method.name, 'seed%d' % seed)
    action_min_max = None

    if cfg.method.name == 'C2FARM':
        explore_replay = c2farm.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            replay_path if cfg.replay.use_disk else None, cams, env,
            cfg.method.voxel_sizes)
        replays = [explore_replay]

        c2farm.launch_utils.fill_replay(
            explore_replay, cfg.rlbench.task, env, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation)

        agent = c2farm.launch_utils.create_agent(cfg, env)

    elif cfg.method.name == 'ARM':
        if len(cams) > 1 or 'front' not in cams:
            raise ValueError('ARM expects only front camera.')
        explore_replay = arm.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            replay_path if cfg.replay.use_disk else None, cams, env)
        replays = [explore_replay]
        all_actions = arm.launch_utils.fill_replay(
            explore_replay, cfg.rlbench.task, env, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams)

        action_min_max = np.min(all_actions, axis=0), np.max(all_actions,
                                                             axis=0)
        action_min_max = _modify_action_min_max(action_min_max)
        agent = arm.launch_utils.create_agent(
            cams[0], cfg.method.activation, cfg.method.q_conf,
            action_min_max, cfg.method.alpha, cfg.method.alpha_lr,
            cfg.method.alpha_auto_tune,
            cfg.method.next_best_pose_critic_lr,
            cfg.method.next_best_pose_actor_lr,
            cfg.method.next_best_pose_critic_weight_decay,
            cfg.method.next_best_pose_actor_weight_decay,
            cfg.method.crop_shape,
            cfg.method.next_best_pose_tau,
            cfg.method.next_best_pose_critic_grad_clip,
            cfg.method.next_best_pose_actor_grad_clip,
            cfg.method.qattention_tau,
            cfg.method.qattention_lr,
            cfg.method.qattention_weight_decay,
            cfg.method.qattention_lambda_qreg,
            env.low_dim_state_len,
            cfg.method.qattention_grad_clip)

    elif cfg.method.name == 'TD3':

        explore_replay = td3.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            replay_path if cfg.replay.use_disk else None, env)
        replays = [explore_replay]

        all_actions = td3.launch_utils.fill_replay(
            explore_replay, cfg.rlbench.task, env, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n)

        action_min_max = np.min(all_actions, axis=0), np.max(all_actions,
                                                             axis=0)
        action_min_max = _modify_action_min_max(action_min_max)
        agent = td3.launch_utils.create_agent(
            cams[0], cfg.method.activation, action_min_max,
            cfg.rlbench.camera_resolution, cfg.method.critic_lr,
            cfg.method.actor_lr, cfg.method.critic_weight_decay,
            cfg.method.actor_weight_decay, cfg.method.tau,
            cfg.method.critic_grad_clip, cfg.method.actor_grad_clip,
            env.low_dim_state_len)

    elif cfg.method.name == 'SAC':

        explore_replay = sac.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            replay_path if cfg.replay.use_disk else None, env)
        replays = [explore_replay]

        all_actions = sac.launch_utils.fill_replay(
            explore_replay, cfg.rlbench.task, env, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n)

        action_min_max = np.min(all_actions, axis=0), np.max(all_actions,
                                                             axis=0)
        # Make translation bounds a little bigger
        action_min_max = _modify_action_min_max(action_min_max)
        agent = sac.launch_utils.create_agent(
            cams[0], cfg.method.activation, action_min_max,
            cfg.rlbench.camera_resolution, cfg.method.critic_lr,
            cfg.method.actor_lr, cfg.method.critic_weight_decay,
            cfg.method.actor_weight_decay, cfg.method.tau,
            cfg.method.critic_grad_clip, cfg.method.actor_grad_clip,
            env.low_dim_state_len, cfg.method.alpha, cfg.method.alpha_auto_tune,
            cfg.method.alpha_lr, cfg.method.decoder_weight_decay,
            cfg.method.decoder_grad_clip, cfg.method.decoder_lr,
            cfg.method.decoder_latent_lambda, cfg.method.encoder_tau)

    elif cfg.method.name == 'DAC':

        replay_demo_path = os.path.join(replay_path, 'demo')
        replay_explore_path = os.path.join(replay_path, 'explore')
        demo_replay = dac.launch_utils.create_replay(
            cfg.replay.batch_size // 2, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            replay_demo_path if cfg.replay.use_disk else None, env)
        explore_replay = dac.launch_utils.create_replay(
            cfg.replay.batch_size // 2, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            replay_explore_path if cfg.replay.use_disk else None, env)
        replays = [demo_replay, explore_replay]
        replay_split = [0.5, 0.5]

        all_actions = dac.launch_utils.fill_replay(
            demo_replay, cfg.rlbench.task, env, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n)

        action_min_max = np.min(all_actions, axis=0), np.max(all_actions,
                                                             axis=0)
        action_min_max = _modify_action_min_max(action_min_max)
        agent = dac.launch_utils.create_agent(
            cams[0], cfg.method.activation, action_min_max,
            cfg.rlbench.camera_resolution, cfg.method.critic_lr,
            cfg.method.actor_lr, cfg.method.critic_weight_decay,
            cfg.method.actor_weight_decay, cfg.method.tau,
            cfg.method.critic_grad_clip, cfg.method.actor_grad_clip,
            env.low_dim_state_len, cfg.method.lambda_gp,
            cfg.method.discriminator_lr, cfg.method.discriminator_grad_clip,
            cfg.method.discriminator_weight_decay)

    elif cfg.method.name == 'bc':
        if train_envs > 0:
            logging.warning('Training envs set to 0 for BC.')
            train_envs = 0
        replay_ratio = None  # No need for replay ratio for BC.
        explore_replay = bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            replay_path if cfg.replay.use_disk else None, env)
        replays = [explore_replay]
        bc.launch_utils.fill_replay(
            explore_replay, cfg.rlbench.task, env, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n)
        agent = bc.launch_utils.create_agent(
            cams[0], cfg.method.activation, cfg.method.lr,
            cfg.method.weight_decay, cfg.rlbench.camera_resolution,
            cfg.method.grad_clip, env.low_dim_state_len)
    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)

    wrapped_replays = [PyTorchReplayBuffer(r) for r in replays]
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')
    logdir = os.path.join(cwd, 'seed%d' % seed)

    if action_min_max is not None:
        # Needed if we want to run the agent again
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, 'action_min_max.pkl'), 'wb') as f:
            pickle.dump(action_min_max, f)

    env_runner = EnvRunner(
        train_env=env, agent=agent, train_replay_buffer=explore_replay,
        num_train_envs=train_envs,
        num_eval_envs=cfg.framework.eval_envs,
        episodes=99999,
        episode_length=cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        env_device=env_device)

    train_runner = PyTorchTrainRunner(
        agent, env_runner,
        wrapped_replays, train_device, replay_split, stat_accum,
        iterations=cfg.framework.training_iterations,
        save_freq=100, log_freq=cfg.framework.log_freq, logdir=logdir,
        weightsdir=weightsdir,
        replay_ratio=replay_ratio,
        transitions_before_train=cfg.framework.transitions_before_train,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging)
    train_runner.start()
    del train_runner
    del env_runner
    torch.cuda.empty_cache()


@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None:
    logging.info('\n' + OmegaConf.to_yaml(cfg))

    train_device = _get_device(cfg.framework.gpu)
    env_device = _get_device(cfg.framework.env_gpu)
    logging.info('Using training device %s.' % str(train_device))
    logging.info('Using env device %s.' % str(env_device))

    action_mode = ActionMode(
        ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME,
        GripperActionMode.OPEN_AMOUNT)

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    if cfg.rlbench.task not in task_files:
        raise ValueError('Task %s not recognised!.' % cfg.rlbench.task)
    task_class = task_file_to_task_class(cfg.rlbench.task)

    cfg.rlbench.cameras = cfg.rlbench.cameras if isinstance(
        cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    obs_config = _create_obs_config(cfg.rlbench.cameras,
                                    cfg.rlbench.camera_resolution)

    env = CustomRLBenchEnv(
        task_class=task_class, observation_config=obs_config,
        action_mode=action_mode, dataset_root=cfg.rlbench.demo_path,
        episode_length=cfg.rlbench.episode_length, headless=True)

    cwd = os.getcwd()
    logging.info('CWD:' + os.getcwd())
    existing_seeds = len(list(filter(lambda x: 'seed' in x, os.listdir(cwd))))

    for seed in range(existing_seeds, existing_seeds + cfg.framework.seeds):
        logging.info('Starting seed %d.' % seed)
        run_seed(cfg, env, cfg.rlbench.cameras, train_device, env_device, seed)


if __name__ == '__main__':
    main()
