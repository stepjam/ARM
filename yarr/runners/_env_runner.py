import copy
import logging
import os
import time
from multiprocessing import Process, Manager
from typing import Any

import numpy as np
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator

# try:
#     if get_start_method() != 'spawn':
#         set_start_method('spawn', force=True)
# except RuntimeError:
#     pass


class _EnvRunner(object):
    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 episodes: int,
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 weightsdir: str = None):
        self._train_env, self._eval_env = train_env, eval_env
        self._agent = agent
        self._train_envs, self._eval_envs = train_envs, eval_envs
        self._episodes, self._episode_length  = episodes, episode_length
        self._rollout_generator = rollout_generator
        self._weightsdir = weightsdir
        self._previous_loaded_weight_folder = ''

        self._timesteps = timesteps

        self._p_args = {}
        self.p_failures = {}
        manager = Manager()
        self.write_lock = manager.Lock()
        self.stored_transitions = manager.list()
        self.agent_summaries = manager.list()
        self._kill_signal = kill_signal
        self._step_signal = step_signal
        self._save_load_lock = save_load_lock
        self._current_replay_ratio = current_replay_ratio
        self._target_replay_ratio = target_replay_ratio

    def restart_process(self, name: str):
        '''
        restart a process to run the environment

        Input:
        - name: the nickname/tag for this environment
        '''

        p = Process(target=self._run_env, args=self._p_args[name], name=name)
        p.start()
        return p

    def spin_up_envs(self, name: str, num_envs: int, eval: bool):
        '''
        create subprocess for running the environment

        Input:
        - name: the nickname/tag for this environment
        - num_envs: number of environment we would like to initiate
        - eval: whether the environment is in `eval` mode

        Output:
        - ps: the subprocesses
        '''

        ps = []
        for i in range(num_envs):
            n = name + str(i)
            self._p_args[n] = (n, eval)
            self.p_failures[n] = 0
            p = Process(target=self._run_env, args=self._p_args[n], name=n)
            p.start()
            ps.append(p)
        return ps

    def _load_save(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # Only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[
                            -1]:
                        self._previous_loaded_weight_folder = weight_folders[
                            -1]
                        d = os.path.join(self._weightsdir,
                                         str(weight_folders[-1]))
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # Rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' %
                                     (self._name, d))
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def _run_env(self, name: str, eval: bool):

        self._name = name
        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=False)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._train_env
        if eval:
            env = self._eval_env
        env.eval = eval
        env.launch()
        for ep in range(self._episodes):
            self._load_save()
            logging.debug('%s: Starting episode %d.' % (name, ep))
            episode_rollout = []
            generator = self._rollout_generator.generator(
                self._step_signal, env, self._agent, self._episode_length,
                self._timesteps, eval)
            try:
                for replay_transition in generator:
                    while True:
                        if self._kill_signal.value:
                            env.shutdown()
                            return
                        if (eval or self._target_replay_ratio is None
                                or self._step_signal.value <= 0
                                or (self._current_replay_ratio.value >
                                    self._target_replay_ratio)):
                            break
                        time.sleep(1)
                        logging.debug(
                            'Agent. Waiting for replay_ratio %f to be more than %f'
                            % (self._current_replay_ratio.value,
                               self._target_replay_ratio))

                    with self.write_lock:
                        if len(self.agent_summaries) == 0:
                            # Only store new summaries if the previous ones
                            # have been popped by the main env runner.
                            for s in self._agent.act_summaries():
                                self.agent_summaries.append(s)
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                env.shutdown()
                raise e

            with self.write_lock:
                for transition in episode_rollout:
                    self.stored_transitions.append((name, transition, eval))
        env.shutdown()

    def kill(self):
        self._kill_signal.value = True
