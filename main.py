import mujoco as mj
import mujoco.viewer as viewer

from move import Movement


def load_model(file_path: str) -> tuple[mj.MjModel, mj.MjData]:
    """ Loads the .xml model into mj.MjModel and mj.MjData structs """

    m = mj.MjModel.from_xml_path(file_path)
    d = mj.MjData(m)

    return m, d


def simulate_move(m: mj.MjModel, d: mj.MjData, move: Movement, view: viewer.Handle = None,
                  debug: dict = None) -> None:
    # disables ALL contact forces
    m.opt.disableflags |= 1

    while move.step(m, d):
        if view and view.is_running():
            view.sync()
            time.sleep(model.opt.timestep)

        if debug:
            debug['qvel_max'].append(data.qvel[3])
            debug['qacc_max'].append(data.qacc[3])

    # re-enables contact forces for future (physics) simulations
    m.opt.disableflags -= 1


if __name__ == '__main__':
    import time

    model, data = load_model('models/xml/humanoid_mesh.xml')
    # move = Movement('data/ACCAD/Female1Running_c3d/C5 - walk to run_poses.npz')
    # move = Movement('data/ACCAD/Female1Walking_c3d/B18 - walk to leap to walk_poses.npz', end=360)
    move = Movement('data/ACCAD/Female1Gestures_c3d/D6- CartWheel_poses.npz', end=270)
    # move = Movement('data/ACCAD/Male2MartialArtsKicks_c3d/G19-  reverse spin cresent left t2_poses.npz', end=150)
    model.opt.timestep = move.timestep

    try:
        view = viewer.launch_passive(model, data)
        debug = False
        time.sleep(1)
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
