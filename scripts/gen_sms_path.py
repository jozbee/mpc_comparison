import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from exp_mpc.stewart_min.spec import TableMPC, state0, get_R, gravity, TableSol
import exp_mpc.stewart_min.robo as robo
from exp_mpc.stewart_min.viz import animate_trajectory
from viz_sms_data import load_sms_data


def extract_dynamics_from_sms(sms_df, resample_dt=None, start_time=40.0):
    """
    Extract linear accelerations and angular velocities from SMS data

    Parameters
    ----------
    sms_df : pandas.DataFrame
        DataFrame containing SMS data
    resample_dt : float, optional
        Time step for resampling the data (seconds)

    Returns
    -------
    tuple
        (timestamps, linear_accelerations, angular_velocities)
    """
    if sms_df is None or sms_df.empty:
        print("No SMS data available")
        return None, None, None

    # Determine the time column
    time_col = (
        "sys.exec.out.time"
        if "sys.exec.out.time" in sms_df.columns
        else "sesmt.md.merged_frame.time"
    )

    # Extract timestamps
    timestamps = sms_df[time_col].values

    # Extract linear accelerations
    linear_acc = np.column_stack(
        [
            sms_df["sesmt.md.merged_frame.xyz_acc[0]"].values,
            sms_df["sesmt.md.merged_frame.xyz_acc[1]"].values,
            sms_df["sesmt.md.merged_frame.xyz_acc[2]"].values,
        ]
    )

    # Extract angular velocities
    angular_vel = np.column_stack(
        [
            sms_df["sesmt.md.merged_frame.ang_vel[0]"].values,
            sms_df["sesmt.md.merged_frame.ang_vel[1]"].values,
            sms_df["sesmt.md.merged_frame.ang_vel[2]"].values,
        ]
    )

    # Filter out data before the specified start time
    start_idx = np.searchsorted(timestamps, start_time)
    if start_idx < len(timestamps):
        timestamps = timestamps[start_idx:]
        linear_acc = linear_acc[start_idx:]
        angular_vel = angular_vel[start_idx:]

    # If resampling is needed
    if resample_dt is not None:
        # Create a new time vector with regular intervals
        t_start = timestamps[0]
        t_end = timestamps[-1]
        new_times = np.arange(t_start, t_end, resample_dt)

        # Resample linear accelerations and angular velocities
        lin_acc_resampled = np.zeros((len(new_times), 3))
        ang_vel_resampled = np.zeros((len(new_times), 3))

        # For each new timestamp, find the closest value in the original data
        for i, t in enumerate(new_times):
            # Find the closest time index
            idx = np.argmin(np.abs(timestamps - t))
            lin_acc_resampled[i] = linear_acc[idx]
            ang_vel_resampled[i] = angular_vel[idx]

        return new_times, lin_acc_resampled, ang_vel_resampled

    return timestamps, linear_acc, angular_vel


def generate_stewart_path_from_sms(
    timestamps, linear_accelerations, angular_velocities, max_duration=None
):
    """
    Generate Stewart platform path from SMS data dynamics

    Parameters
    ----------
    timestamps : numpy.ndarray
        Array of timestamps
    linear_accelerations : numpy.ndarray
        Array of linear accelerations, shape (n, 3)
    angular_velocities : numpy.ndarray
        Array of angular velocities, shape (n, 3)
    max_duration : float, optional
        Maximum duration to simulate (seconds)

    Returns
    -------
    tuple
        (waypoints, mpc_linear_accelerations, mpc_angular_velocities)
    """
    # Initialize the MPC controller
    mpc = TableMPC.create_default()

    # Set weights to prioritize tracking reference dynamics
    # mpc.set_weights(w_a=1e2, w_omega=1e2, w_leg=1e0, w_control=1e-1)
    mpc.set_weights(w_acc_x=1e1, w_omega=5e2, w_leg=1e2, w_control=1e-1)

    # Limit the duration if specified
    if max_duration is not None:
        max_idx = np.searchsorted(timestamps, timestamps[0] + max_duration)
        if max_idx < len(timestamps):
            timestamps = timestamps[:max_idx]
            linear_accelerations = linear_accelerations[:max_idx]
            angular_velocities = angular_velocities[:max_idx]

    # Calculate number of simulation steps
    num_steps = len(timestamps)
    print(
        f"Simulating {num_steps} steps over "
        f"{timestamps[-1] - timestamps[0]:.2f} seconds"
    )

    # Initialize state (starting from the default state0)
    current_state = state0.copy()

    # Initialize storage for waypoints, accelerations, and angular velocities
    waypoints = []
    mpc_linear_accelerations = []
    mpc_angular_velocities = []
    solutions: list[TableSol] = []

    # Run simulation
    for i in tqdm.tqdm(range(num_steps)):
        # Get reference dynamics for current time step
        a_ref = linear_accelerations[i].copy()
        omega_ref = angular_velocities[i].copy()

        # Set reference for this time step
        mpc.set_reference(a_ref=a_ref, omega_ref=omega_ref)

        # Solve MPC from current state
        mpc.solve(current_state)

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

        # Store accelerations and angular velocities from MPC solution
        mpc_linear_accelerations.append(mpc.control_sol[0, :3].copy())

        # Calculate angular velocities from the state using the same approach as gen_stewart_path.py
        phi = current_state[3]
        theta = current_state[4]
        phi_dot = current_state[9]
        theta_dot = current_state[10]
        psi_dot = current_state[11]

        # Calculating angular velocities in body frame
        omega_x = phi_dot - np.sin(theta) * psi_dot
        omega_y = (
            np.cos(phi) * theta_dot + np.sin(phi) * np.cos(theta) * psi_dot
        )
        omega_z = (
            -np.sin(phi) * theta_dot + np.cos(phi) * np.cos(theta) * psi_dot
        )

        mpc_angular_velocities.append([omega_x, omega_y, omega_z])

        # Use MPC prediction for next state
        current_state = mpc.state_sol[1].copy()

        # Store the solution for later analysis
        solutions.append(mpc.get_solution())

    # Timing statistics
    times = [sol.stats.time for sol in solutions]
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    print(
        f"Average solve time: {avg_time:.4f} ± {std_time:.4f} ms "
        f"(min: {min_time:.4f}, max: {max_time:.4f})"
    )

    return (
        waypoints,
        np.array(mpc_linear_accelerations),
        np.array(mpc_angular_velocities),
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
        ca_R = get_R(phi, theta, psi)
        R = np.empty(ca_R.shape)
        for j in range(ca_R.shape[0]):
            for k in range(ca_R.shape[1]):
                R[j, k] = ca_R[j, k]

        # Transform acceleration to head frame
        acc_world = linear_accelerations[i]
        acc_head = R.T @ (acc_world + gravity)

        head_frame_accelerations.append(acc_head)

    return np.array(head_frame_accelerations)


def plot_sms_vs_mpc_dynamics(
    timestamps,
    sms_linear_acc,
    sms_angular_vel,
    mpc_linear_acc,
    mpc_angular_vel,
    waypoints,
):
    """
    Plot SMS data dynamics versus MPC generated dynamics

    Parameters
    ----------
    timestamps : numpy.ndarray
        Array of timestamps
    sms_linear_acc : numpy.ndarray
        SMS linear accelerations
    sms_angular_vel : numpy.ndarray
        SMS angular velocities
    mpc_linear_acc : numpy.ndarray
        MPC generated linear accelerations
    mpc_angular_vel : numpy.ndarray
        MPC generated angular velocities
    waypoints : list
        List of waypoints generated by MPC
    """
    # Transform MPC accelerations to head frame for comparison with SMS data
    mpc_head_acc = transform_to_head_frame(waypoints, mpc_linear_acc)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # Linear accelerations comparison
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(timestamps, sms_linear_acc[:, 0], "r--", label="SMS X")
    ax1.plot(timestamps, sms_linear_acc[:, 1], "g--", label="SMS Y")
    ax1.plot(timestamps, sms_linear_acc[:, 2], "b--", label="SMS Z")
    ax1.plot(timestamps, mpc_head_acc[:, 0], "r-", label="MPC X")
    ax1.plot(timestamps, mpc_head_acc[:, 1], "g-", label="MPC Y")
    ax1.plot(timestamps, mpc_head_acc[:, 2], "b-", label="MPC Z")
    ax1.set_title("Linear Accelerations Comparison (Head Frame)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Acceleration [m/s²]")
    ax1.legend()
    ax1.grid(True)

    # Angular velocities comparison
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(timestamps, sms_angular_vel[:, 0], "r--", label="SMS ω_x")
    ax2.plot(timestamps, sms_angular_vel[:, 1], "g--", label="SMS ω_y")
    ax2.plot(timestamps, sms_angular_vel[:, 2], "b--", label="SMS ω_z")
    ax2.plot(timestamps, mpc_angular_vel[:, 0], "r-", label="MPC ω_x")
    ax2.plot(timestamps, mpc_angular_vel[:, 1], "g-", label="MPC ω_y")
    ax2.plot(timestamps, mpc_angular_vel[:, 2], "b-", label="MPC ω_z")
    ax2.set_title("Angular Velocities Comparison (Body Frame)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angular Velocity [rad/s]")
    ax2.legend()
    ax2.grid(True)

    # Individual linear acceleration comparisons for clarity
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(timestamps, sms_linear_acc[:, 0], "r--", label="SMS")
    ax3.plot(timestamps, mpc_head_acc[:, 0], "r-", label="MPC")
    ax3.set_title("X Acceleration Comparison")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Acceleration [m/s²]")
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(timestamps, sms_linear_acc[:, 1], "g--", label="SMS")
    ax4.plot(timestamps, mpc_head_acc[:, 1], "g-", label="MPC")
    ax4.set_title("Y Acceleration Comparison")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Acceleration [m/s²]")
    ax4.legend()
    ax4.grid(True)

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(timestamps, sms_linear_acc[:, 2], "b--", label="SMS")
    ax5.plot(timestamps, mpc_head_acc[:, 2], "b-", label="MPC")
    ax5.set_title("Z Acceleration Comparison")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Acceleration [m/s²]")
    ax5.legend()
    ax5.grid(True)

    # Platform position plot
    waypoints_array = np.array(waypoints)
    ax6 = fig.add_subplot(3, 2, 6, projection="3d")
    ax6.plot(
        waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2]
    )
    ax6.set_title("Platform 3D Trajectory")
    ax6.set_xlabel("X [m]")
    ax6.set_ylabel("Y [m]")
    ax6.set_zlabel("Z [m]")  # type: ignore
    ax6.grid(True)

    # Create output directory for plots if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the figure
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sms_mpc_comparison_{timestamp}.png"))

    return fig


if __name__ == "__main__":
    # Define the path to the SMS data file
    file_path = "/Users/jozbee/work/eng/comp/data/00_sms_drive.csv"

    # Load the SMS data
    sms_df = load_sms_data(file_path)

    # Extract dynamics from SMS data
    # resample_dt = 0.05  # 50ms between samples
    resample_dt = None
    timestamps, sms_linear_acc, sms_angular_vel = extract_dynamics_from_sms(
        sms_df, resample_dt
    )
    assert timestamps is not None, "Failed to extract timestamps from SMS data"
    assert sms_linear_acc is not None, (
        "Failed to extract linear accelerations from SMS data"
    )
    assert sms_angular_vel is not None, (
        "Failed to extract angular velocities from SMS data"
    )

    # We need to negate the z-axis reference
    # (Lower gravity should mean that that the table drops)
    moon_gravity = 1.625
    sms_linear_acc[:, 2] = sms_linear_acc[:, 2] + 2 * moon_gravity

    # Limit the duration for faster processing during development
    max_duration = 40.0  # seconds

    # Generate Stewart platform path from SMS dynamics
    # The accelerations in SMS data are in head frame
    waypoints, mpc_linear_acc, mpc_angular_vel = generate_stewart_path_from_sms(
        timestamps,
        sms_linear_acc,
        sms_angular_vel,
        max_duration,
    )

    # Plot comparison between SMS and MPC dynamics
    fig_comparison = plot_sms_vs_mpc_dynamics(
        timestamps[: len(waypoints)],  # Use same length for all arrays
        sms_linear_acc[: len(waypoints)],
        sms_angular_vel[: len(waypoints)],
        mpc_linear_acc,
        mpc_angular_vel,
        waypoints,
    )

    # Visualize the platform motion using the 3D visualizer
    print(f"Generated {len(waypoints)} waypoints.")
    robo_params = robo.RoboParams()
    robo_geom = robo.RoboGeom()
    anim, fig_viz = animate_trajectory(
        trajectory=waypoints,
        sim_rate=0.5,
        fps=30,
        robo_params=robo_params,
        robo_geom=robo_geom,
    )

    print("Visualization complete. Displaying plots.")
    plt.show()
