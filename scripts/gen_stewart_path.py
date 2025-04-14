import tqdm
import numpy as np
import matplotlib.pyplot as plt

from exp_mpc.stewart_min.spec import TableMPC, state0, dt, get_R, gravity
from exp_mpc.stewart_min.viz import visualize_trajectory


def generate_constant_acceleration_path(
    acceleration: float = 0.05,
    duration: float = 5.0,
) -> tuple[list[list[float]], np.ndarray, np.ndarray]:
    """Generate a path with constant acceleration in the x direction using MPC.

    Parameters
    ----------
    acceleration :
        Desired acceleration in x direction (m/s^2)
    duration :
        Total duration of the path (seconds)

    Returns
    -------
    waypoints, linear_accelerations, angular_velocities
    """
    # mpc init
    mpc = TableMPC.create_default()
    mpc.set_weights(w_a=1e1, w_omega=1e1, w_leg=1e2, w_control=1e-1)

    # set reference acceleration in x direction
    a_ref = np.array([acceleration, 0.0, 0.0]) + gravity
    mpc.set_reference(a_ref=a_ref)

    # bookeeping
    waypoints = []
    linear_accelerations = []
    angular_velocities = []

    # run simulation
    current_state = state0.copy()
    num_steps = int(duration / dt)
    for _ in tqdm.tqdm(range(num_steps)):
        mpc.solve(current_state)

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

        # don't forget to copy (otherwise, everything will be the same)
        linear_accelerations.append(mpc.control_sol[0, :3].copy())

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

        current_state = mpc.sim_next_state()

    return (
        waypoints,
        np.array(linear_accelerations),
        np.array(angular_velocities),
    )


def transform_to_head_frame(waypoints, linear_accelerations):
    """Transform linear accelerations from world frame to head frame."""
    head_frame_accelerations = []

    for i, waypoint in enumerate(waypoints):
        # Extract the orientation from the waypoint
        phi = waypoint[3]  # roll
        theta = waypoint[4]  # pitch
        psi = waypoint[5]  # yaw

        # Create rotation matrix
        ca_R = get_R(phi, theta, psi)  # Convert to numpy array
        R = np.empty(ca_R.shape)
        for i in range(ca_R.shape[0]):
            for j in range(ca_R.shape[1]):
                R[i, j] = ca_R[i, j]

        # Transform acceleration to head frame (similar to get_acceleration in spec.py)
        acc_world = linear_accelerations[i]
        acc_head = R.T @ (acc_world + gravity)

        head_frame_accelerations.append(acc_head)

    return np.array(head_frame_accelerations)


def plot_dynamics(linear_accelerations, angular_velocities):
    """Plot linear accelerations and angular velocities."""
    time = np.arange(len(linear_accelerations)) * dt

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

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

    fig.tight_layout()
    return fig


def plot_dynamics_with_head_frame(
    waypoints,
    linear_accelerations,
    angular_velocities,
    a_ref=np.array([0.0, 0.0, 0.0]),
    omega_ref=np.array([0.0, 0.0, 0.0]),
):
    """Plot linear accelerations and angular velocities in both world and head frames.

    Parameters
    ----------
    waypoints : list of lists
        List of waypoints, each containing [x, y, z, roll, pitch, yaw]
    linear_accelerations : ndarray
        Array of linear accelerations in world frame
    angular_velocities : ndarray
        Array of angular velocities in body frame
    a_ref : ndarray, shape (3,)
        Reference linear acceleration in head frame [x, y, z]
    omega_ref : ndarray, shape (3,)
        Reference angular velocity in body frame [ωx, ωy, ωz]
    """
    # Input validation
    a_ref = np.asarray(a_ref)
    omega_ref = np.asarray(omega_ref)

    assert a_ref.shape == (3,), f"a_ref must have shape (3,), got {a_ref.shape}"
    assert omega_ref.shape == (3,), (
        f"omega_ref must have shape (3,), got {omega_ref.shape}"
    )

    time = np.arange(len(linear_accelerations)) * dt

    # Transform accelerations to head frame
    head_frame_accelerations = transform_to_head_frame(
        waypoints, linear_accelerations
    )

    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))

    # World frame accelerations
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(time, linear_accelerations[:, 0], label="X")
    ax1.plot(time, linear_accelerations[:, 1], label="Y")
    ax1.plot(time, linear_accelerations[:, 2], label="Z")
    ax1.set_title("World Frame Linear Accelerations")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Acceleration [m/s²]")
    ax1.legend()
    ax1.grid(True)

    # Head frame accelerations
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(time, head_frame_accelerations[:, 0], color="r", label="X")
    ax2.plot(time, head_frame_accelerations[:, 1], color="g", label="Y")
    ax2.plot(time, head_frame_accelerations[:, 2], color="b", label="Z")
    # Use a_ref directly as it's already in head frame
    ax2.axhline(y=a_ref[0], color="r", linestyle="--", label="X ref")
    ax2.axhline(y=a_ref[1], color="g", linestyle="--", label="Y ref")
    ax2.axhline(y=a_ref[2], color="b", linestyle="--", label="Z ref")
    ax2.set_title("Head Frame Linear Accelerations")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Acceleration [m/s²]")
    ax2.legend()
    ax2.grid(True)

    # Angular velocities (body frame)
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(time, angular_velocities[:, 0], color="r", label="ω_x")
    ax3.plot(time, angular_velocities[:, 1], color="g", label="ω_y")
    ax3.plot(time, angular_velocities[:, 2], color="b", label="ω_z")
    ax3.axhline(y=omega_ref[0], color="r", linestyle="--", label="ω_x ref")
    ax3.axhline(y=omega_ref[1], color="g", linestyle="--", label="ω_y ref")
    ax3.axhline(y=omega_ref[2], color="b", linestyle="--", label="ω_z ref")
    ax3.set_title("Angular Velocities (Body Frame)")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Angular Velocity [rad/s]")
    ax3.legend()
    ax3.grid(True)

    # Individual plots for clarity
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(time, head_frame_accelerations[:, 0], "r")
    ax4.axhline(y=a_ref[0], color="r", linestyle="--", label="X ref")
    ax4.set_title("Head Frame X Acceleration")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Acceleration [m/s²]")
    ax4.legend()
    ax4.grid(True)

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(time, head_frame_accelerations[:, 1], "g")
    ax5.axhline(y=a_ref[1], color="g", linestyle="--", label="Y ref")
    ax5.set_title("Head Frame Y Acceleration")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Acceleration [m/s²]")
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(time, head_frame_accelerations[:, 2], "b")
    ax6.axhline(y=a_ref[2], color="b", linestyle="--", label="Z ref")
    ax6.set_title("Head Frame Z Acceleration")
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Acceleration [m/s^2]")
    ax6.legend()
    ax6.grid(True)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Parameters
    acceleration = 1.0  # m/s^2
    duration = 10.0  # seconds

    # Generate waypoints, accelerations, and angular velocities
    waypoints, linear_accelerations, angular_velocities = (
        generate_constant_acceleration_path(
            acceleration=acceleration,
            duration=duration,
        )
    )

    # Plot the dynamics in world frame
    fig_world = plot_dynamics(linear_accelerations, angular_velocities)
    plt.savefig("stewart_dynamics_world_frame.png")

    # Plot the dynamics in both world and head frames
    a_ref = np.array([acceleration, 0.0, 0.0]) + gravity
    omega_ref = np.zeros(3)  # Reference angular velocity (zero)

    fig_both = plot_dynamics_with_head_frame(
        waypoints,
        linear_accelerations,
        angular_velocities,
        a_ref=a_ref,
        omega_ref=omega_ref,
    )
    plt.savefig("stewart_dynamics_head_frame.png")

    # Visualize the platform motion using the 3D visualizer
    print(f"Generated {len(waypoints)} waypoints.")
    anim, fig_viz = visualize_trajectory(waypoints, dt=dt, sim_rate=1.0, fps=30)

    plt.show()
