import os

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle

from move import Movement

DEFAULT_SIZE = 480


class HumanoidEnv(MujocoEnv, EzPickle):
    metadata = {
        'render_modes': [
            'human',
            'rgb_array',
            'depth_array',
        ],
        'render_fps': 120,
    }

    def __init__(self, model_path: str = 'assets/humanoid.xml', data_path: str = 'data',
                 frame_skip: int = 5, **kwargs):
        self.metadata['render_fps'] = int(np.round(self.metadata['render_fps'] / frame_skip))

        # Hard-coded values for the number of values in qpos (nv = 91) and qvel (nu = 69).
        obs_space = gym.spaces.Dict({
            'agent': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(91 + 69,), dtype=np.float64),
            'target': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(91 + 69,), dtype=np.float64),
        })

        MujocoEnv.__init__(
            self, os.path.abspath(model_path), frame_skip, observation_space=obs_space, **kwargs
        )
        EzPickle.__init__(self, **kwargs)
        self.target = mj.MjData(self.model)

        self.movements = [os.path.abspath(os.path.join(root, name))
                          for root, _, files in os.walk(data_path) for name in files if name.endswith('.npz')]
        self._initialize_movement()
        try:
            self.mujoco_renderer.render(render_mode=kwargs['render_mode'])
        except KeyError:
            pass

    def _initialize_movement(self):
        # Choose new motion to imitate and initialize the models to its first pose.
        while True:
            try:
                self.motion = Movement(data_file=self.movements[np.random.randint(0, len(self.movements))])
                break
            except KeyError:
                pass
        self.motion.set_position(self.model, self.data)
        self.motion.set_position(self.model, self.target)

    def _reset_simulation(self):
        MujocoEnv._reset_simulation(self)
        mj.mj_resetData(self.model, self.target)
        self._initialize_movement()

        return self._get_obs()

    def _get_obs(self):
        return {
            'agent': np.append(self.data.qpos, self.data.qvel),
            'target': np.append(self.target.qpos, self.target.qvel)
        }

    def step(self, action):
        self._step_mujoco_simulation(action, n_frames=self.frame_skip)

        # Position of target motion at times t - 1 and t.
        qpos_old = np.append(*self.motion.get_action())
        self.motion.curr += 1
        center_motion, quat_motion = self.motion.get_action()
        rot_motion = R.from_quat(np.reshape(quat_motion, newshape=(-1, 4)))

        # Resulting position of the humanoid after the action given by the agent.
        center_agent = self.data.qpos[: 3]
        rot_agent = R.from_quat(np.reshape(self.data.qpos[3:], newshape=(-1, 4)))

        # Compute the reward function.
        # Taken from DeepMimic (https://dl.acm.org/doi/pdf/10.1145/3197517.3201311)

        # The pose reward which encourages the joint orientations to match the reference.
        rot_diff = rot_agent * R.inv(rot_motion)
        r_p = np.exp(-2 * np.sum(np.linalg.norm(rot_diff.as_rotvec(), ord=2, axis=1) ** 2))

        # The velocity reward encourages each joint's angular velocity to match the reference.
        qvel_motion = np.zeros(self.model.nv)
        mj.mj_differentiatePos(
            self.model, qvel_motion, self.dt, qpos_old, np.append(center_motion, quat_motion)
        )
        r_v = np.exp(-0.1 * np.sum(
            np.linalg.norm(np.reshape(self.data.qvel - qvel_motion, newshape=(-1, 3)), ord=2, axis=1) ** 2
        ))

        # The end-effector reward encourages the humanoid's hands and feet to match the reference.
        # The index of the left, right feet and left, right hands are 10, 11, 22, 23 respectively.
        r_e = np.exp(-40 * np.sum(
            np.linalg.norm(np.take(self.data.xpos - self.target.xpos, [10, 11, 20, 21], axis=0), ord=2, axis=1) ** 2
        ))

        # The center-of-mass reward penalizes deviations in the humanoid's center-of-mass.
        r_c = np.exp(-10 * np.linalg.norm(center_motion - center_agent, ord=2))

        # Calculate the overall value of the reward function.
        w_p, w_v, w_e, w_c = 0.65, 0.1, 0.15, 0.1
        reward = w_p * r_p + w_v * r_v + w_e * r_e + w_c * r_c

        info = {
            'reward_pose': r_p,
            'reward_vel': r_v,
            'reward_end': r_e,
            'reward_center': r_c,
        }

        truncated = bool(np.sum(self.data.warning.number[4: 7]))

        # Returns the observation, reward, terminated, truncated, and information.
        return self._get_obs(), reward, self.motion.curr >= self.motion.end, truncated, info

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        )

        return self._get_obs()
