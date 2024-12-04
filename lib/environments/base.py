import gymnasium as gym
from abc import ABC, abstractmethod


class BaseEnvironment(ABC, gym.Env):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, action):
        pass

    def _get_info(self):
        return {}


