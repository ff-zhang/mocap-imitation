import os

import numpy as np
import scipy
import mujoco as mj
import mujoco.viewer as viewer


def load_model(file_path: str) -> tuple[mj.MjModel, mj.MjData]:
    """
    Loads the .xml model into mujoco
    :param file_path
    :return: tuple of the model definition and its initial state
    """
    model = mj.MjModel.from_xml_path(file_path)
    data = mj.MjData(model)

    return model, data


def load_smpl_data(file_path: str) -> tuple[np.array, np.array]:
    """
    Loads SMPL pose and translation data from a .npz file
    :param file_path
    :return: tuple of pose and translation data
    """

    assert file_path[-4:] == '.npz'
    if not os.path.isfile(file_path):
        raise FileNotFoundError

    data = np.load(file_path)
    joint_data = data['poses'][:, :22 * 3]

    return joint_data.reshape((joint_data.shape[0], joint_data.shape[1] // 3, 3)), data['trans']


class Movement:
    """"""
    def __init__(self, data_file: str) -> None:
        # self.model = load_model(model_path)
        # self.data = load_smpl_data(data_path)

        self.pose, self.trans = load_smpl_data(data_file)

        self.init_pos = self.pose[0]
        self.count = 0
        self.num_frames = self.pose.shape[0]

        self.end = self.count == self.num_frames - 1

    def get_next_action(self):
        self.count += 1
        if self.count == self.num_frames - 1:
            self.end = True

        return self.pose[self.count]


if __name__ == '__main__':
    model, data = load_model('models/xml/humanoid_mesh.xml')
    move = Movement('data/ACCAD/Female1Running_c3d/C2 - Run to stand_poses.npz')

    try:
        view = viewer.launch_passive(model, data)
    except RuntimeError:
        view = None

    import time

    # for i in range(24):
    #     data.ctrl[i] = i + 1

    t1 = np.copy(data.qpos)

    while not move.end:
        act = move.get_next_action()

        # print(data.qpos)
        # data.qpos[:] = act

        """
        data.qpos breakdown
              [:3] - (x, y, z) Cartesian coordinate of the pelvis 
            [3:99] - (w, x, y, z) rotational quaterion of each part relative to its parent 
        """
        data.qpos[:] = t1

        mj.mj_step(model, data)
        if view:
            view.sync()

        time.sleep(0.01)
