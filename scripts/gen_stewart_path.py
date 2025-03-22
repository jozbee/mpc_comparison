import typing as tp

import numpy as np
import matplotlib.pyplot as plt

from exp_mpc.stewart_min.spec import TableMPC, state0, n, dt
from exp_mpc.stewart_min.viz import visualize_trajectory


def generate_constant_acceleration_path(
    acceleration: float = 0.05, duration: float = 5.0, dt_sim: float = 0.1
) -> tuple[list[list[float]], np.ndarray, np.ndarray]:
    """Generate a path with constant acceleration in the x direction using MPC.

    Parameters
    ----------
    acceleration :
        Desired acceleration in x direction (m/s²)
    duration :
        Total duration of the path (seconds)
    dt_sim :
        Time step for simulation (seconds)

    Returns
    -------
    waypoints, linear_accelerations, angular_velocities
    """
    # Initialize the MPC controller
    mpc = TableMPC.create_default()

    # Set weights to prioritize smooth motion
    mpc.set_weights(w_a=10.0, w_omega=10.0, w_leg=1.0, w_control=0.1)

    # Set reference acceleration in x direction
    a_ref = np.array([acceleration, 0.0, -9.81])
    mpc.set_reference(a_ref=a_ref)

    # Number of simulation steps
    num_steps = int(duration / dt_sim)

    # Initialize state (starting from the default state0)
    current_state = state0.copy()

    # Initialize storage for waypoints, accelerations, and angular velocities
    waypoints = []
    linear_accelerations = []
    angular_velocities = []

    # Run simulation
    for _ in range(num_steps):
        # Solve MPC from current state
        mpc.solve(current_state)
        # if _ % 100 == 0:
        #     mpc.plot_solution().show()
        #     input()

        # Store the current pose as a waypoint [x, y, z, roll, pitch, yaw]
        waypoints.append(
            [
                current_state[0],  # x
                current_state[1],  # y
                current_state[2],  # z
                current_state[3],  # roll (phi)
                current_state[4],  # pitch (theta)
                current_state[5],  # yaw (psi)
            ]
        )

        # Store the current control inputs (accelerations)
        # Don't forget to copy (otherwise, everything will be the same)
        linear_accelerations.append(mpc.control_sol[0, :3].copy())

        # Calculate angular velocities from the state
        phi = current_state[3]
        theta = current_state[4]
        # psi = current_state[5]
        phi_dot = current_state[9]
        theta_dot = current_state[10]
        psi_dot = current_state[11]

        # Calculating angular velocities in body frame using the PHI matrix
        # This is equivalent to the get_angular_velocity function in spec.py
        omega_x = phi_dot - np.sin(theta) * psi_dot
        omega_y = (
            np.cos(phi) * theta_dot + np.sin(phi) * np.cos(theta) * psi_dot
        )
        omega_z = (
            -np.sin(phi) * theta_dot + np.cos(phi) * np.cos(theta) * psi_dot
        )

        angular_velocities.append([omega_x, omega_y, omega_z])

        # Use MPC prediction for next state without manual modifications
        current_state = mpc.state_sol[1].copy()

    return (
        waypoints,
        np.array(linear_accelerations),
        np.array(angular_velocities),
    )


def plot_dynamics(linear_accelerations, angular_velocities, dt_sim):
    """Plot linear accelerations and angular velocities."""
    time = np.arange(len(linear_accelerations)) * dt_sim

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Plot linear accelerations
    axs[0, 0].plot(time, linear_accelerations[:, 0])
    axs[0, 0].set_title("X Acceleration")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Acceleration (m/s²)")

    axs[0, 1].plot(time, linear_accelerations[:, 1])
    axs[0, 1].set_title("Y Acceleration")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Acceleration (m/s²)")

    axs[0, 2].plot(time, linear_accelerations[:, 2])
    axs[0, 2].set_title("Z Acceleration")
    axs[0, 2].set_xlabel("Time (s)")
    axs[0, 2].set_ylabel("Acceleration (m/s²)")

    # Plot angular velocities
    axs[1, 0].plot(time, angular_velocities[:, 0])
    axs[1, 0].set_title("Roll Rate (ω_x)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Angular Velocity (rad/s)")

    axs[1, 1].plot(time, angular_velocities[:, 1])
    axs[1, 1].set_title("Pitch Rate (ω_y)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Angular Velocity (rad/s)")

    axs[1, 2].plot(time, angular_velocities[:, 2])
    axs[1, 2].set_title("Yaw Rate (ω_z)")
    axs[1, 2].set_xlabel("Time (s)")
    axs[1, 2].set_ylabel("Angular Velocity (rad/s)")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Parameters
    acceleration = 0.05  # m/s²
    duration = 5.0  # seconds

    # Generate waypoints, accelerations, and angular velocities
    waypoints, linear_accelerations, angular_velocities = (
        generate_constant_acceleration_path(
            acceleration=acceleration, duration=duration, dt_sim=dt
        )
    )

    # Plot the dynamics (accelerations and angular velocities)
    fig = plot_dynamics(linear_accelerations, angular_velocities, dt)
    plt.savefig("stewart_dynamics.png")
    plt.show()

    # Visualize the platform motion using the 3D visualizer
    print(f"Generated {len(waypoints)} waypoints.")
    anim, fig_viz = visualize_trajectory(waypoints, dt=dt, sim_rate=0.5, fps=30)

    plt.show()
