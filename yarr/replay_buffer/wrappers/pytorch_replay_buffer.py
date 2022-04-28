import time
from threading import Lock, Thread

from torch.utils.data import IterableDataset, DataLoader

from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.replay_buffer.wrappers import WrappedReplayBuffer


class PyTorchIterableReplayDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer):
        self._replay_buffer = replay_buffer

    def _generator(self):
        while True:
            yield self._replay_buffer.sample_transition_batch(
                pack_in_dict=True)

    def __iter__(self):
        return iter(self._generator())


class PyTorchReplayBuffer(WrappedReplayBuffer):
    """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

    Usage:
      To add a transition:  call the `add` function.

      To sample a batch:    Construct operations that depend on any of the \\
                            tensors is the transition dictionary. Every \\
                            sess.run that requires any of these tensors will \\
                            sample a new transition.
    """
    def __init__(self, replay_buffer: ReplayBuffer, num_workers: int = 2):
        super(PyTorchReplayBuffer, self).__init__(replay_buffer)
        self._num_workers = num_workers

    def dataset(self) -> DataLoader:
        # d = PyTorchIterableReplayDataset(self._replay_buffer, self._num_workers)
        d = PyTorchIterableReplayDataset(self._replay_buffer)
        # Batch size None disables automatic batching
        return DataLoader(d, batch_size=None, pin_memory=True)
