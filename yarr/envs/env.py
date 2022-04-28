from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition


class Env(ABC):

    def __init__(self):
        self._eval_env = False

    @property
    def eval(self):
        return self._eval_env

    @eval.setter
    def eval(self, eval):
        self._eval_env = eval

    @abstractmethod
    def launch(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> dict:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Transition:
        pass

    @property
    @abstractmethod
    def observation_elements(self) -> List[ObservationElement]:
        pass

    @property
    @abstractmethod
    def action_shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def env(self) -> Any:
        pass
