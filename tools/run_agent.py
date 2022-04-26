import numpy as np
import torch
from absl import app, flags
from hydra.experimental import initialize, compose
from moviepy.editor import *
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from omegaconf import OmegaConf, ListConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class

from arm import c2farm, qte, lpr
from arm.custom_rlbench_env import CustomRLBenchEnv
from arm.lpr.trajectory_action_mode import TrajectoryActionMode
from launch import _create_obs_config
from tools.utils import RLBenchCinematic

FREEZE_DURATION = 2
FPS = 20

flags.DEFINE_string('logdir', '/path/to/log/dir', 'weight dir.')
flags.DEFINE_string('method', 'C2FARM', 'The method to run.')
flags.DEFINE_string('task', 'take_lid_off_saucepan', 'The task to run.')
flags.DEFINE_integer('episodes', 1, 'The number of episodes to run.')

FLAGS = flags.FLAGS


def _save_clips(clips, name):
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile('%s.mp4' % name)


def visualise(logdir, task, method):
    config_path = os.path.join(logdir, task, method, '.hydra')
    weights_path = os.path.join(logdir, task, method, 'seed0', 'weights')

    if not os.path.exists(config_path):
        raise ValueError('No cofig in: ' + config_path)
    if not os.path.exists(weights_path):
        raise ValueError('No weights in: ' + weights_path)

    with initialize(config_path=os.path.relpath(config_path)):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))

    cfg.rlbench.cameras = cfg.rlbench.cameras if isinstance(
        cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]

    obs_config = _create_obs_config(
        cfg.rlbench.cameras, cfg.rlbench.camera_resolution)
    task_class = task_file_to_task_class(task)

    gripper_mode = Discrete()
    if cfg.method.name == 'PathARM':
        arm_action_mode = TrajectoryActionMode(cfg.method.trajectory_points)
    else:
        arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    env = CustomRLBenchEnv(
        task_class=task_class, observation_config=obs_config,
        action_mode=action_mode, dataset_root=cfg.rlbench.demo_path,
        episode_length=cfg.rlbench.episode_length, headless=True,
        time_in_state=True)
    _ = env.observation_elements

    if cfg.method.name == 'C2FARM':
        agent = c2farm.launch_utils.create_agent(
            cfg, env, cfg.rlbench.scene_bounds,
            cfg.rlbench.camera_resolution)
    elif cfg.method.name == 'C2FARM+QTE':
        agent = qte.launch_utils.create_agent(
            cfg, env, cfg.rlbench.scene_bounds,
            cfg.rlbench.camera_resolution)
    elif cfg.method.name == 'LPR':
        agent = lpr.launch_utils.create_agent(
            cfg, env, cfg.rlbench.scene_bounds, cfg.rlbench.camera_resolution,
            cfg.method.trajectory_point_noise, cfg.method.trajectory_points,
            cfg.method.trajectory_mode, cfg.method.trajectory_samples)
    else:
        raise ValueError('Invalid method name.')

    agent.build(training=False, device=torch.device("cpu"))
    weight_folders = sorted(map(int, os.listdir(weights_path)))
    agent.load_weights(os.path.join(weights_path, str(weight_folders[-1])))

    env.launch()
    cinemtaic_cam = RLBenchCinematic()
    env.register_callback(cinemtaic_cam.callback)
    for ep in range(FLAGS.episodes):
        obs = env.reset()
        agent.reset()
        obs_history = {
            k: [np.array(v, dtype=_get_type(v))] * cfg.replay.timesteps for
            k, v in obs.items()}
        clips = []
        last = False
        for step in range(cfg.rlbench.episode_length):
            prepped_data = {k: torch.FloatTensor([v]) for k, v in obs_history.items()}
            act_result = agent.act(step, prepped_data, deterministic=True)
            transition = env.step(act_result)

            trajectory_frames = cinemtaic_cam.frames
            if len(trajectory_frames) > 0:
                cinemtaic_cam.empty()
                clips.append(ImageSequenceClip(trajectory_frames, fps=FPS))

            if last:
                break
            if transition.terminal:
                last = True
            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)
        _save_clips(clips, '%s_%s.mp4' % (method, task))

    print('Shutting down env...')
    env.shutdown()


def _get_type(x):
    if x.dtype == np.float64:
        return np.float32
    return x.dtype


def main(argv):
    del argv
    visualise(FLAGS.logdir, FLAGS.task, FLAGS.method)


if __name__ == '__main__':
    app.run(main)
