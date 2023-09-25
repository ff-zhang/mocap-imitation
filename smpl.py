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
    def __init__(self, data_file: str, start: int = 0, joints: list[int] = None) -> None:
        self.pose, self.trans = load_smpl_data(data_file)

        self.init_pos = self.pose[0]
        self.count = start
        self.num_frames = self.pose.shape[0]

        self.end = self.count == self.num_frames - 1

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
        self.joints = joints if joints else [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]

    def get_next_action(self) -> np.array:
        self.count += 1
        if self.count == self.num_frames - 1:
            self.end = True

        quats = np.array(([]))
        for i in self.joints:
            r = scipy.spatial.transform.Rotation.from_rotvec(self.pose[self.count][i]).as_quat()
            quats = np.append(quats, np.append(r[3], r[:3]))

        return self.trans[self.count], quats

    def set_initial_position(self, m: mj.MjModel, d: mj.MjData) -> None:
        center, quats = self.get_next_action()
        d.qpos[: 3] = center
        d.qpos[3: 3 + len(quats)] = quats

        mj.mj_step(m, d)
        d.qvel[: len(d.qvel)] = np.array([0] * len(d.qvel))
        d.qacc[: len(d.qacc)] = np.array([0] * len(d.qacc))

        return None

    def step(self, m: mj.MjModel, d: mj.MjData) -> None:
        """
        data.qpos breakdown
              [:3] - (x, y, z) Cartesian coordinate of the pelvis
            [3:99] - (w, x, y, z) rotational quaterion of each part relative to its parent
        """
        center, quats = move.get_next_action()

        # TODO: needed to prevent clipping into the ground, remove when issue is solved
        d.qpos[: 3] = center + [0, 0, 0.2]
        d.qpos[3: 3 + len(quats)] = quats

        mj.mj_step(m, d)

        if self.end:
            print('Reached of the movement. Resetting sequence.')
            self.count = 0


def simulate_move(m: mj.MjModel, d: mj.MjData, move: Movement,
                  view: viewer.Handle = None, debug: dict = None) -> None:
    # disables ALL contact forces to prevent blow-ups in acceleration due to clipping
    m.opt.disableflags |= 1

    while not move.end:
        move.step(m, d)

        if view and view.is_running():
            view.sync()
            time.sleep(model.opt.timestep)

        if debug:
            debug['qvel_max'].append(data.qvel[3])
            debug['qacc_max'].append(data.qacc[3])

    # re-enables contact forces for future (physics) simulations
    model.opt.disableflags -= 1


if __name__ == '__main__':
    import time

    model, data = load_model('models/xml/humanoid_mesh.xml')
    move = Movement('data/ACCAD/Female1Running_c3d/C5 - walk to run_poses.npz', start=500)

    try:
        view = viewer.launch_passive(model, data)
        debug = False
        time.sleep(2)
    except RuntimeError:
        view = None
        debug = True

    if debug:
        debug = {'qvel_max': [], 'qacc_max': []}

    move.set_initial_position(model, data)
    simulate_move(model, data, move, view, debug)

    start = time.time()
    while time.time() - start < 2.0:
        mj.mj_step(model, data)

        if view and view.is_running():
            view.sync()
            time.sleep(model.opt.timestep)

        if debug:
            debug['qvel_max'].append(max(data.qvel))
            debug['qacc_max'].append(max(data.qacc))

    if view and view.is_running():
        view.close()

    if debug:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title('Max Velocity')
        plt.plot(debug['qvel_max'])
        plt.xlabel(r'Time ($t$)')
        plt.ylabel(r'Velocity')
        plt.show()

        plt.figure()
        plt.title('Max Acceleration')
        plt.plot(debug['qacc_max'])
        plt.xlabel(r'Time ($t$)')
        plt.ylabel(r'Acceleration')
        plt.show()
