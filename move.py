import os

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Slerp, Rotation as R


class Movement:
    def __init__(self, data_file: str, m: mj.MjModel, start: int = 0, end: int = np.inf,
                 offset: list[float] = (0., 0., 0.3)) -> None:
        """
        Relationship between joints and their labelling:
            Pelvis [0]
              |  L_Hip [1]
              |    |  L_Knee [4]
              |    |    |  L_Ankle [7]
              |    |----|----|- L_Toe [10]
              |  R_Hip [2]
              |    |  R_Knee [5]
              |    |    |  R_Ankle [8]
              |    |----|----|- R_Toe [11]
              |  Torso [3]
              |    |  Spine [6]
              |    |    |  Chest [9]
              |    |    |    |  Neck [12]
              |    |    |    |    |- Head [15]
              |    |    |    |  L_Thorax [13]
              |    |    |    |    |  L_Shoulder [16]
              |    |    |    |    |    |  L_Elbow [18]
              |    |    |    |    |    |    |  L_Wrist [20]
              |    |    |    |    |----|----|----|- L_Hand [22]
              |    |    |    |  R_Thorax [14]
              |    |    |    |    |  R_Shoulder [17]
              |    |    |    |    |    |  R_Elbow [19]
              |    |    |    |    |    |    |  R_Wrist [21]
              |----|----|----|----|----|----|----|- R_Hand [23]

        The order of the joints in this list is IMPORTANT.
        """
        self.joints = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
        self.offset = offset
        self.model = m

        self.pose, self.trans, self.timestep = self._load_smpl(data_file)
        self.num_frames = self.pose.shape[0]

        # Rotations and slerps have the same order as the joints.
        self.rots = [R.from_rotvec(self.pose[:, j, :]) for j in self.joints]
        self.slerp = [Slerp(self.timestep * np.arange(self.num_frames), rot) for rot in self.rots]

        self.start, self.end = start, min(end, self.num_frames)
        self.curr = self.start

        self._init_frames()

    def _load_smpl(self, file_path: str) -> tuple[np.array, np.array, float]:
        """ Loads SMPL pose and translation data from a .npz file """

        assert file_path[-4:] == '.npz'
        if not os.path.isfile(file_path):
            raise FileNotFoundError

        data = np.load(file_path)
        self.pose = data['poses'][:, : 3 * self.model.njnt]
        self.pose = self.pose.reshape((self.pose.shape[0], self.pose.shape[1] // 3, 3))

        return self.pose, data['trans'], 1 / data['mocap_framerate']

    def _init_frames(self) -> None:
        self.qpos = np.array([
            np.roll(np.array([slerp(t).as_quat() for slerp in self.slerp]), shift=1, axis=1).reshape(-1)
            for t in self.timestep * np.arange(self.num_frames)])
        self.qpos = np.hstack([self.trans, self.qpos])

        self.qvel = np.zeros(shape=(self.num_frames, self.model.nv))
        for t in np.arange(self.start + 1, self.end):
            mj.mj_differentiatePos(self.model, self.qvel[t], self.timestep, self.qpos[t - 1], self.qpos[t])

        self.qacc = (self.qvel - np.roll(self.qvel, shift=1, axis=0)) / self.timestep
        self.qacc[0] = 0.

    def set_movement(self, m: mj.MjModel, d: mj.MjData) -> None:
        d.qpos = self.qpos[self.curr]
        d.qvel = self.qvel[self.curr]
        d.qacc = self.qacc[self.curr]

    def step(self, m: mj.MjModel, d: mj.MjData) -> bool:
        if self.curr == self.end:
            print('Reached of the movement.')
            return False

        self.set_movement(m, d)
        self.curr += 1
        mj.mj_forward(m, d)

        return True
