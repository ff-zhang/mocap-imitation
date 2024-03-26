import numpy as np
import mujoco as mj

import matplotlib.pyplot as plt


NUM_JOINTS = 22

_debug = {
    'qvel': [],
    'qacc': [],
    'qfrc': [],
    'subtree_vel': [],
    'subtree_acc': [],
    'subtree_frc': [],
    'subtree_trq': [],
}


def debug_capture(m: mj.MjModel, d: mj.MjData) -> None:
    _debug['qvel'].append(get_qvel(d))
    _debug['qacc'].append(get_qacc(d))
    _debug['qfrc'].append(get_qfrc(m, d))
    _debug['subtree_vel'].append(get_subtree_vel(d))
    _debug['subtree_acc'].append(get_subtree_acc(d))
    _debug['subtree_frc'].append(get_subtree_frc(d))
    _debug['subtree_trq'].append(get_subtree_trq(d))


def debug_plot(save: bool = True, dir_path: str = 'figures/') -> None:
    print("Saving/showing figures saved for debugging.")
    for k in _debug.keys():
        _debug[k] = np.moveaxis(np.array(_debug[k]), 0, 1)
        fig, axs = plt.subplots(_debug[k].shape[0], sharex='all', figsize=(15, 60))
        for i in range(_debug[k].shape[0]):
            axs[i].plot(_debug[k][i])
        plt.tight_layout()
        plt.show() if not save else plt.savefig(dir_path + f'{k}.png')


def get_names(m: mj.MjModel) -> dict[str, int]:
    return {name: i for name, i in enumerate(m.names.decode('utf-8').split('\x00')[2:])}


def get_mass(m: mj.MjModel) -> np.array:
    return np.copy(m.body_mass[1:])


def get_qpos(d: mj.MjData) -> np.array:
    return np.copy(d.qpos)


def get_qvel(d: mj.MjData) -> np.array:
    return np.copy(d.qvel)


def get_qacc(d: mj.MjData) -> np.array:
    return np.copy(d.qacc)


def get_qfrc(m: mj.MjModel, d: mj.MjData) -> np.array:
    mj.mj_inverse(m, d)
    return np.copy(d.qfrc_inverse)


def get_subtree_vel(d: mj.MjData) -> np.array:
    return np.copy(d.sensordata[0: 3 * NUM_JOINTS]).reshape((NUM_JOINTS, 3))


def get_subtree_acc(d: mj.MjData) -> np.array:
    return np.copy(d.sensordata[3 * NUM_JOINTS: 6 * NUM_JOINTS]).reshape((NUM_JOINTS, 3))


def get_subtree_frc(d: mj.MjData) -> np.array:
    return np.copy(d.sensordata[6 * NUM_JOINTS: 9 * NUM_JOINTS]).reshape((NUM_JOINTS, 3))


def get_subtree_trq(d: mj.MjData) -> np.array:
    return np.copy(d.sensordata[9 * NUM_JOINTS: 12 * NUM_JOINTS]).reshape((NUM_JOINTS, 3))


if __name__ == '__main__':
    from main import load_model
    from move import Movement

    model, data = load_model('assets/humanoid_mesh.xml')
    move = Movement('data/ACCAD/Female1Gestures_c3d/D6- CartWheel_poses.npz', end=270)
    model.opt.timestep = move.timestep

    move.set_movement(model, data)
    move.step(model, data)
