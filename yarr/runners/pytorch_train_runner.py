import copy
import logging
import os
import shutil
import signal
import sys
import threading
import time
from multiprocessing import Lock
from typing import Optional, List
from typing import Union

import numpy as np
import psutil
import torch
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners.train_runner import TrainRunner
from yarr.utils.log_writer import LogWriter
from yarr.utils.stat_accumulator import StatAccumulator

NUM_WEIGHTS_TO_KEEP = 10


class PyTorchTrainRunner(TrainRunner):
    def __init__(self,
                 agent: Agent,
                 env_runner: EnvRunner,
                 wrapped_replay_buffer: Union[PyTorchReplayBuffer,
                                              List[PyTorchReplayBuffer]],
                 train_device: torch.device,
                 replay_buffer_sample_rates: List[float] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(1e6),
                 logdir: str = '/tmp/yarr/logs',
                 log_freq: int = 10,
                 transitions_before_train: int = 1000,
                 weightsdir: str = '/tmp/yarr/weights',
                 save_freq: int = 100,
                 replay_ratio: Optional[float] = None,
                 tensorboard_logging: bool = True,
                 csv_logging: bool = False):
        super(PyTorchTrainRunner,
              self).__init__(agent, env_runner, wrapped_replay_buffer,
                             stat_accumulator, iterations, logdir, log_freq,
                             transitions_before_train, weightsdir, save_freq)

        env_runner.log_freq = log_freq
        env_runner.target_replay_ratio = replay_ratio
        self._wrapped_buffer = wrapped_replay_buffer if isinstance(
            wrapped_replay_buffer, list) else [wrapped_replay_buffer]
        self._replay_buffer_sample_rates = ([1.0] if
                                            replay_buffer_sample_rates is None
                                            else replay_buffer_sample_rates)
        if len(self._replay_buffer_sample_rates) != len(wrapped_replay_buffer):
            raise ValueError(
                'Numbers of replay buffers differs from sampling rates.')
        if sum(self._replay_buffer_sample_rates) != 1:
            raise ValueError('Sum of sampling rates should be 1.')

        self._train_device = train_device
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging

        if replay_ratio is not None and replay_ratio < 0:
            raise ValueError("max_replay_ratio must be positive.")
        self._target_replay_ratio = replay_ratio

        self._writer = None
        if logdir is None:
            logging.info("'logdir' was None. No logging will take place.")
        else:
            self._writer = LogWriter(self._logdir, tensorboard_logging,
                                     csv_logging)
        if weightsdir is None:
            logging.info(
                "'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)

    def _save_model(self, i):
        '''
        save the weight of the agent/policy

        Input:
        - i: the step index for weight saving
        '''

        with self._save_load_lock:
            d = os.path.join(self._weightsdir, str(i))
            os.makedirs(d, exist_ok=True)
            self._agent.save_weights(d)
            # Remove oldest save
            prev_dir = os.path.join(
                self._weightsdir,
                str(i - self._save_freq * NUM_WEIGHTS_TO_KEEP))
            if os.path.exists(prev_dir):
                shutil.rmtree(prev_dir)

    def _step(self, i, sampled_batch):
        '''
        update step for the agent with the sample batch transition
        
        Input:
        - i:
        - sampled_batch: 
        '''

        update_dict = self._agent.update(i, sampled_batch)
        acc_bs = 0
        for wb in self._wrapped_buffer:
            bs = wb.replay_buffer.batch_size
            if 'priority' in update_dict:
                wb.replay_buffer.set_priority(
                    sampled_batch['indices'][acc_bs:acc_bs +
                                             bs].cpu().detach().numpy(),
                    update_dict['priority'][acc_bs:acc_bs + bs])
            acc_bs += bs

    def _signal_handler(self, sig, frame):
        if threading.current_thread().name != 'MainThread':
            return
        logging.info('SIGINT captured. Shutting down.'
                     'This may take a few seconds.')
        self._env_runner.stop()
        [r.replay_buffer.shutdown() for r in self._wrapped_buffer]
        sys.exit(0)

    def _get_add_counts(self):
        return np.array(
            [r.replay_buffer.add_count for r in self._wrapped_buffer])

    def _get_sum_add_counts(self):
        return sum([r.replay_buffer.add_count for r in self._wrapped_buffer])

    def start(self):
        '''
        start the RL learning process
        '''

        signal.signal(signal.SIGINT, self._signal_handler)

        self._save_load_lock = Lock()

        # Kick off the environments
        self._env_runner.start(self._save_load_lock)

        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._train_device)

        if self._weightsdir is not None:
            self._save_model(0)  # Save weights so workers can load.

        while (np.any(
                self._get_add_counts() < self._transitions_before_train)):
            time.sleep(1)
            logging.info(
                'Waiting for %d samples before training. Currently have %s.' %
                (self._transitions_before_train, str(self._get_add_counts())))

        datasets = [r.dataset() for r in self._wrapped_buffer]
        data_iter = [iter(d) for d in datasets]

        init_replay_size = self._get_sum_add_counts().astype(float)
        batch_size = sum(
            [r.replay_buffer.batch_size for r in self._wrapped_buffer])
        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()

        for i in range(self._iterations):
            self._env_runner.set_step(i)

            log_iteration = i % self._log_freq == 0 and i > 0

            if log_iteration:
                process.cpu_percent(interval=None)

            def get_replay_ratio():
                size_used = batch_size * i
                size_added = (self._get_sum_add_counts() - init_replay_size)
                replay_ratio = size_used / (size_added + 1e-6)
                return replay_ratio

            if self._target_replay_ratio is not None:
                # wait for env_runner collecting enough samples
                while True:
                    replay_ratio = get_replay_ratio()
                    self._env_runner.current_replay_ratio.value = replay_ratio
                    if replay_ratio < self._target_replay_ratio:
                        break
                    time.sleep(1)
                    logging.debug(
                        'Waiting for replay_ratio %f to be less than %f.' %
                        (replay_ratio, self._target_replay_ratio))
                del replay_ratio

            t = time.time()
            sampled_batch = [next(di) for di in data_iter]

            if len(sampled_batch) > 1:
                result = {}
                for key in sampled_batch[0]:
                    result[key] = torch.cat([d[key] for d in sampled_batch], 0)
                sampled_batch = result
            else:
                sampled_batch = sampled_batch[0]

            sample_time = time.time() - t
            batch = {
                k: v.to(self._train_device)
                for k, v in sampled_batch.items()
            }
            t = time.time()
            self._step(i, batch)
            step_time = time.time() - t

            if log_iteration and self._writer is not None:
                replay_ratio = get_replay_ratio()
                logging.info(
                    'Step %d. Sample time: %s. Step time: %s. Replay ratio: %s.'
                    % (i, sample_time, step_time, replay_ratio))
                agent_summaries = self._agent.update_summaries()
                env_summaries = self._env_runner.summaries()
                self._writer.add_summaries(i, agent_summaries + env_summaries)

                for r_i, wrapped_buffer in enumerate(self._wrapped_buffer):
                    self._writer.add_scalar(
                        i, 'replay%d/add_count' % r_i,
                        wrapped_buffer.replay_buffer.add_count)
                    self._writer.add_scalar(
                        i, 'replay%d/size' % r_i,
                        wrapped_buffer.replay_buffer.replay_capacity
                        if wrapped_buffer.replay_buffer.is_full() else
                        wrapped_buffer.replay_buffer.add_count)

                self._writer.add_scalar(i, 'replay/replay_ratio', replay_ratio)
                self._writer.add_scalar(
                    i, 'replay/update_to_insert_ratio',
                    float(i) / float(self._get_sum_add_counts() -
                                     init_replay_size + 1e-6))

                self._writer.add_scalar(i, 'monitoring/sample_time_per_item',
                                        sample_time / batch_size)
                self._writer.add_scalar(i, 'monitoring/train_time_per_item',
                                        step_time / batch_size)
                self._writer.add_scalar(i, 'monitoring/memory_gb',
                                        process.memory_info().rss * 1e-9)
                self._writer.add_scalar(
                    i, 'monitoring/cpu_percent',
                    process.cpu_percent(interval=None) / num_cpu)

            self._writer.end_iteration()

            if i % self._save_freq == 0 and self._weightsdir is not None:
                self._save_model(i)

        if self._writer is not None:
            self._writer.close()

        logging.info('Stopping envs ...')
        self._env_runner.stop()
        [r.replay_buffer.shutdown() for r in self._wrapped_buffer]
