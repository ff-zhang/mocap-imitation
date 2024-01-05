import os

import numpy as np
import mujoco as mj

import gymnasium as gym
from gymnasium.spaces import Space
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle

from stable_baselines3 import PPO

DEFAULT_SIZE = 480


class HumanoidEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, model_path: str = 'assets/humanoid.xml', frame_skip: int = 5, **kwargs):
        # Hard-coded values for the number of values in qpos and qvel.
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(91 + 69,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self, os.path.abspath(model_path), frame_skip, observation_space=obs_space, **kwargs
        )
        EzPickle.__init__(self, **kwargs)

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def step(self, action):
        pass

    def reset_model(self):
        pass

    def viewer_setup(self):
        assert self.viewer is not None


if __name__ == '__main__':
    env = HumanoidEnv()

    env.close()
