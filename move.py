import os

import numpy as np
import scipy
import mujoco as mj


class Movement:
    def __init__(self, data_file: str, start: int = 0, end: int = np.inf,
                 offset: list[float] = (0., 0., 0.), joints: list[int] = None) -> None:
        self.pose, self.trans, self.timestep = self.load_smpl(data_file)
        self.num_frames = self.pose.shape[0]

        self.curr = start
        self.end = min(end, self.num_frames - 1)

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
        """
        self.joints = joints if joints else [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16,
                                             18, 20, 14, 17, 19, 21]
        self.offset = offset

    def load_smpl(self, file_path: str) -> tuple[np.array, np.array, float]:
        """ Loads SMPL pose and translation data from a .npz file """

        assert file_path[-4:] == '.npz'
        if not os.path.isfile(file_path):
            raise FileNotFoundError

        data = np.load(file_path)
        self.pose = data['poses'][:, :22 * 3]
        self.pose = self.pose.reshape((self.pose.shape[0], self.pose.shape[1] // 3, 3))

        return self.pose, data['trans'], 1 / data['mocap_framerate']

    def get_next_action(self) -> np.array:
        self.curr += 1

        quaternion = np.array(([]))
        for i in self.joints:
            r = scipy.spatial.transform.Rotation.from_rotvec(self.pose[self.curr][i]).as_quat()
            quaternion = np.append(quaternion, np.append(r[3], r[:3]))

        return self.trans[self.curr], quaternion

    def set_initial_position(self, m: mj.MjModel, d: mj.MjData) -> None:
        center, quaternion = self.get_next_action()
        d.qpos[: 3] = center + self.offset
        d.qpos[3: 3 + len(quaternion)] = quaternion

        mj.mj_step(m, d)
        d.qvel[: len(d.qvel)] = np.array([0] * len(d.qvel))
        d.qacc[: len(d.qacc)] = np.array([0] * len(d.qacc))

    def step(self, m: mj.MjModel, d: mj.MjData) -> bool:
        if self.curr == self.end:
            print('Reached of the movement.')
            return False

        """
        data.qpos breakdown
              [:3] - (x, y, z) Cartesian coordinate of the pelvis
            [3:99] - (w, x, y, z) rotational quaterion of each part relative to its parent
        """
        center, quaternion = self.get_next_action()
        mj.mj_differentiatePos(m, d.qvel, m.opt.timestep, d.qpos,
                               np.append(center + self.offset, quaternion))

        d.qpos[: 3] = center + self.offset
        d.qpos[3: 3 + len(quaternion)] = quaternion

        mj.mj_forward(m, d)

        return True
