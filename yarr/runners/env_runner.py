import collections
import logging
import os
import signal
import time
from multiprocessing import Value
from threading import Thread
from typing import List
from typing import Union

import numpy as np

from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.runners._env_runner import _EnvRunner
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import StatAccumulator


class EnvRunner(object):
    def __init__(self,
                 env: Env,
                 agent: Agent,
                 replay_buffer: ReplayBuffer,
                 train_envs: int,
                 eval_envs: int,
                 episodes: int,
                 episode_length: int,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 rollout_generator: RolloutGenerator = None,
                 weightsdir: str = None,
                 max_fails: int = 10):
        self._env, self._agent = env, agent
        self._train_envs, self._eval_envs = train_envs, eval_envs
        self._replay_buffer = replay_buffer
        self._episodes = episodes
        self._episode_length = episode_length
        self._stat_accumulator = stat_accumulator
        self._rollout_generator = (RolloutGenerator()
                                   if rollout_generator is None else
                                   rollout_generator)
        self._weightsdir = weightsdir
        self._max_fails = max_fails
        self._previous_loaded_weight_folder = ''
        self._p = None
        self._kill_signal = Value('b', 0)
        self._step_signal = Value('i', -1)
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        self._total_transitions = {'train_envs': 0, 'eval_envs': 0}
        self.log_freq = 1000  # Will get overridden later
        self.target_replay_ratio = None  # Will get overridden later
        self.current_replay_ratio = Value('f', -1)

    def summaries(self) -> List[Summary]:
        summaries = []
        if self._stat_accumulator is not None:
            summaries.extend(self._stat_accumulator.pop())
        for key, value in self._new_transitions.items():
            summaries.append(ScalarSummary('%s/new_transitions' % key, value))
        for key, value in self._total_transitions.items():
            summaries.append(ScalarSummary('%s/total_transitions' % key,
                                           value))
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        summaries.extend(self._agent_summaries)
        return summaries

    def _update(self):
        '''
        Move the stored transitions to the replay and accumulate statistics.
        '''
        
        new_transitions = collections.defaultdict(int)
        with self._internal_env_runner.write_lock:
            self._agent_summaries = list(
                self._internal_env_runner.agent_summaries)
            if self._step_signal.value % self.log_freq == 0 and self._step_signal.value > 0:
                self._internal_env_runner.agent_summaries[:] = []
            for name, transition, eval in self._internal_env_runner.stored_transitions:
                if not eval:
                    kwargs = dict(transition.observation)
                    self._replay_buffer.add(np.array(transition.action),
                                            transition.reward,
                                            transition.terminal,
                                            transition.timeout, **kwargs)
                    if transition.terminal:
                        self._replay_buffer.add_final(
                            **transition.final_observation)
                new_transitions[name] += 1
                self._new_transitions[
                    'eval_envs' if eval else 'train_envs'] += 1
                self._total_transitions[
                    'eval_envs' if eval else 'train_envs'] += 1
                if self._stat_accumulator is not None:
                    self._stat_accumulator.step(transition, eval)
            self._internal_env_runner.stored_transitions[:] = []  # Clear list
        return new_transitions

    def _run(self, save_load_lock):
        self._internal_env_runner = _EnvRunner(
            self._env, self._env, self._agent, self._replay_buffer.timesteps,
            self._train_envs, self._eval_envs, self._episodes,
            self._episode_length, self._kill_signal, self._step_signal,
            self._rollout_generator, save_load_lock, self.current_replay_ratio,
            self.target_replay_ratio, self._weightsdir)
        training_envs = self._internal_env_runner.spin_up_envs(
            'train_env', self._train_envs, False)
        eval_envs = self._internal_env_runner.spin_up_envs(
            'eval_env', self._eval_envs, True)
        envs = training_envs + eval_envs
        no_transitions = {env.name: 0 for env in envs}
        while True:
            for p in envs:
                if p.exitcode is not None:
                    envs.remove(p)
                    if p.exitcode != 0:
                        self._internal_env_runner.p_failures[p.name] += 1
                        n_failures = self._internal_env_runner.p_failures[
                            p.name]
                        if n_failures > self._max_fails:
                            logging.error(
                                'Env %s failed too many times (%d times > %d)'
                                % (p.name, n_failures, self._max_fails))
                            raise RuntimeError('Too many process failures.')
                        logging.warning(
                            'Env %s failed (%d times <= %d). restarting' %
                            (p.name, n_failures, self._max_fails))
                        p = self._internal_env_runner.restart_process(p.name)
                        envs.append(p)

            if not self._kill_signal.value:
                new_transitions = self._update()
                for p in envs:
                    if new_transitions[p.name] == 0:
                        no_transitions[p.name] += 1
                    else:
                        no_transitions[p.name] = 0
                    if no_transitions[p.name] > 600:  # 5min
                        logging.warning("Env %s hangs, so restarting" % p.name)
                        envs.remove(p)
                        os.kill(p.pid, signal.SIGTERM)
                        p = self._internal_env_runner.restart_process(p.name)
                        envs.append(p)
                        no_transitions[p.name] = 0

            if len(envs) == 0:
                break
            time.sleep(1)

    def start(self, save_load_lock):
        self._p = Thread(target=self._run,
                         args=(save_load_lock, ),
                         daemon=True)
        self._p.name = 'EnvRunnerThread'
        self._p.start()

    def wait(self):
        if self._p.is_alive():
            self._p.join()

    def stop(self):
        if self._p.is_alive():
            self._kill_signal.value = True
            self._p.join()

    def set_step(self, step):
        self._step_signal.value = step
