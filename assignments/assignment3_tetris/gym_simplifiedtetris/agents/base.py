"""Base agent class."""

from abc import ABC, abstractmethod
from typing import Any

import gym
import numpy as np


class BaseAgent(ABC):
    """Agent base class."""

    @abstractmethod
    def predict(self, obs: np.ndarray, env: gym.Env, **kwargs: Any) -> int:
        raise NotImplementedError
