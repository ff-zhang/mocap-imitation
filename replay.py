import mujoco as mj
import mujoco.viewer as viewer

import utils
from move import Movement


def load_model(file_path: str) -> tuple[mj.MjModel, mj.MjData]:
    """ Loads the .xml model into mj.MjModel and mj.MjData structs """

    m = mj.MjModel.from_xml_path(file_path)
    d = mj.MjData(m)

    return m, d


def simulate_move(m: mj.MjModel, d: mj.MjData, move: Movement, view: viewer.Handle = None,
                  debug: bool = False) -> None:
    # disables ALL contact forces
    # m.opt.disableflags |= 1

    while move.step(m, d):
        if view and view.is_running():
            view.sync()
            time.sleep(model.opt.timestep)

        if debug:
            utils.debug_capture(model, data)

    # re-enables contact forces for future (physics) simulations
    # m.opt.disableflags -= 1


if __name__ == '__main__':
    import time

    data_files = [
        'data/ACCAD/Female1Running_c3d/C5 - walk to run_poses.npz',
        'data/ACCAD/Female1Walking_c3d/B18 - walk to leap to walk_poses.npz',
        'data/ACCAD/Female1Gestures_c3d/D6- CartWheel_poses.npz',
        'data/ACCAD/Male2MartialArtsKicks_c3d/G19-  reverse spin cresent left t2_poses.npz',
    ]

    model, data = load_model('assets/humanoid.xml')
    move = Movement(data_files[0], model)
    # move = Movement(data_files[1], model, end=360)
    # move = Movement(data_files[2], model, end=270, offset=[0., 0., 0.1])
    model.opt.timestep = move.timestep

    try:
        view = viewer.launch_passive(model, data)
        debug = False
        time.sleep(1)
    except RuntimeError:
        view = None
        debug = True

    move.set_movement(model, data)
    move.curr += 1

    simulate_move(model, data, move, view, debug)

    start = time.time()
    while time.time() - start < 2.0:
        mj.mj_step(model, data)

        if view and view.is_running():
            view.sync()
            time.sleep(model.opt.timestep)

        if debug:
            utils.debug_capture(model, data)

    if view and view.is_running():
        view.close()

    if debug:
        utils.debug_plot()
