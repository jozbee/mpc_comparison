import tqdm
import numpy as np
import matplotlib.pyplot as plt

import exp_mpc.stewart_min.spec as spec
import exp_mpc.stewart_min.viz as viz


def generate_constant_acceleration_path(
    a_ref: np.ndarray = np.array([1.0, 0.0, 0.0]) + spec.gravity,
    duration: float = 5.0,
) -> list[spec.TableSol]:
    """Generate a path with constant acceleration in the x direction using MPC.

    Parameters
    ----------
    a_ref :
        Desired acceleration in x direction [m/s^2].
    duration :
        Total duration of the path [s].

    Returns
    -------
    MPC solutions
    """
    # mpc init
    mpc = spec.TableMPC.create_default()
    mpc.set_weights(w_a=1e1, w_omega=1e1, w_leg=1e2, w_control=1e-1)

    # set reference acceleration in x direction
    a_ref = np.array([acceleration, 0.0, 0.0]) + spec.gravity
    mpc.set_reference(a_ref=a_ref)

    # bookeeping
    solutions = []

    # run simulation
    current_state = spec.state0.copy()
    num_steps = int(duration / spec.dt)
    for _ in tqdm.tqdm(range(num_steps)):
        mpc.solve(current_state)
        solutions.append(mpc.get_solution())
        current_state = mpc.sim_next_state()

    return solutions


if __name__ == "__main__":
    # parameters
    acceleration = 1.0  # m/s^2
    duration = 10.0  # seconds
    a_ref = np.array([acceleration, 0.0, 0.0]) + spec.gravity

    # generate waypoints, accelerations, and angular velocities
    solutions = generate_constant_acceleration_path(
        a_ref=a_ref,
        duration=duration,
    )

    # plot dynamics
    omega_ref = np.zeros(3)  # reference angular velocity (zero)
    references = {
        "xyz-acceleration": np.tile(a_ref, reps=(len(solutions), 1)),
        "angular-velocity": np.tile(omega_ref, reps=(len(solutions), 1)),
    }
    fig_head = viz.plot_human_trajectory(solutions, references)
    fig_table = viz.plot_cartesian_table_trajectory(solutions)

    # visualize the platform motion using the 3D visualizer
    anim, fig_viz = viz.animate_trajectory(solutions, sim_rate=1.0, fps=30)

    # plot everything
    plt.show()
