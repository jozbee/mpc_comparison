import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from exp_mpc.stewart_min.spec import bots, tops, leg_min, leg_max

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def compute_leg_lengths(position, rotation_matrix):
    """
    Compute leg lengths for a given position and orientation of the platform.

    Args:
        position: [x, y, z] position of the platform center
        rotation_matrix: 3x3 rotation matrix for platform orientation

    Returns:
        leg_lengths: array of 6 leg lengths
    """
    leg_lengths = []

    for i in range(6):
        # Transform platform attachment point to world coordinates
        top_i = rotation_matrix @ tops[i] + position

        # Compute leg vector
        leg_vector = top_i - bots[i]

        # Compute leg length
        leg_length = np.linalg.norm(leg_vector)
        leg_lengths.append(leg_length)

    return np.array(leg_lengths)


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    Compute rotation matrix from roll, pitch, yaw angles (in radians).
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )

    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    R_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    R = R_z @ R_y @ R_x
    return R


def visualize_trajectory(trajectory, dt=0.02, sim_rate=1.0, fps=30):
    """Visualize the Stewart platform following a trajectory.

    Parameters
    ----------
    trajectory : array-like
        List of waypoints, where each waypoint is a list or array of
        [x, y, z, roll, pitch, yaw] defining the platform's position and
        orientation.
    dt : float, optional
        Time step between simulation frames in seconds.
    sim_rate : float, optional
        Simulation rate multiplier (1.0 = real-time, 2.0 = twice as fast).
    fps : int, optional
        Frames per second for the animation.
    Returns
    -------
    matplotlib.animation.FuncAnimation and matplotlib.figure.Figure
        Animation object that can be further processed (e.g., saved to file),
        and its corresponding figure.
    """
    # Setup figure
    fig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[1.0, 0.1],
        width_ratios=[1.0, 0.7],
        wspace=0.35,
        hspace=0.35,
    )

    # 3D plot for platform visualization
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")  # type: ignore
    # ax_3d.set_title('Stewart Platform Visualization')

    # Create a margin around the workspace
    x_lims = np.array(
        [point[0] for point in bots] + [point[0] for point in tops]
    )
    x_min = np.min(x_lims) - 0.5
    x_max = np.max(x_lims) + 0.5
    y_lims = np.array(
        [point[1] for point in bots] + [point[1] for point in tops]
    )
    y_min = np.min(y_lims) - 0.5
    y_max = np.max(y_lims) + 0.5
    z_lims = np.array(
        [point[2] for point in bots] + [point[2] for point in tops]
    )
    z_min = np.min(z_lims) - 0.5
    z_max = np.max(z_lims) + 1.0  # for extra height

    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
    ax_3d.set_zlim(z_min, z_max)  # type: ignore

    # Plot the progression of time
    ax_time = fig.add_subplot(gs[1, 1])
    ax_time.set_xlabel("Progress")
    # ax_time.set_ylabel('Time')
    # ax_time.set_title('Trajectory Progress')
    ax_time.set_xlim(0, 1)
    ax_time.set_ylim(0, 1)
    ax_time.set_yticks([])  # Hide y-ticks for cleaner look
    ax_time.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_time.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    progress_bar = ax_time.barh(0.5, 0, height=0.3, color="blue", alpha=0.7)
    time_text = ax_time.text(0.02, 0.02, "", transform=ax_time.transAxes)

    # Plot for leg lengths
    ax_legs = fig.add_subplot(gs[0, 1])
    ax_legs.set_xlabel("Leg Number")
    ax_legs.set_ylabel("Length (m)")
    # ax_legs.set_title('Leg Lengths')
    ax_legs.set_xlim(0.5, 6.5)
    ax_legs.set_xticks(range(1, 7))
    ax_legs.set_ylim(leg_min * 0.9, leg_max * 1.1)
    ax_legs.axhline(
        y=leg_min, color="r", linestyle="-", alpha=0.5, label="Min Length"
    )
    ax_legs.axhline(
        y=leg_max, color="r", linestyle="-", alpha=0.5, label="Max Length"
    )

    # Initialize visualization elements
    (base_polygon,) = ax_3d.plot([], [], [], "ko-", linewidth=2, markersize=5)
    (platform_polygon,) = ax_3d.plot(
        [], [], [], "bo-", linewidth=2, markersize=5
    )
    legs = [ax_3d.plot([], [], [], "g-", linewidth=1)[0] for _ in range(6)]
    leg_bars = ax_legs.bar(range(1, 7), [0] * 6, color="black", alpha=0.7)
    leg_text = ax_legs.text(
        0.02, 0.95, "", transform=ax_legs.transAxes, verticalalignment="top"
    )

    # Connect base points to form a polygon
    base_x = [bots[i][0] for i in range(6)] + [bots[0][0]]
    base_y = [bots[i][1] for i in range(6)] + [bots[0][1]]
    base_z = [bots[i][2] for i in range(6)] + [bots[0][2]]
    base_polygon.set_data(base_x, base_y)
    base_polygon.set_3d_properties(base_z)  # type: ignore

    # fig.tight_layout()

    # Interpolate trajectory for smoother animation
    num_points = len(trajectory)
    t_values = np.arange(0, dt * num_points, dt)
    t_interp = np.arange(0, dt * num_points, 1.0 / fps * sim_rate)

    # Interpolate positions (x, y, z)
    positions = np.array([point[:3] for point in trajectory])
    x_interp = np.interp(t_interp, t_values, positions[:, 0])
    y_interp = np.interp(t_interp, t_values, positions[:, 1])
    z_interp = np.interp(t_interp, t_values, positions[:, 2])

    # Interpolate orientations (roll, pitch, yaw)
    orientations = np.array([point[3:] for point in trajectory])
    roll_interp = np.interp(t_interp, t_values, orientations[:, 0])
    pitch_interp = np.interp(t_interp, t_values, orientations[:, 1])
    yaw_interp = np.interp(t_interp, t_values, orientations[:, 2])

    interp_trajectory = list(
        zip(x_interp, y_interp, z_interp, roll_interp, pitch_interp, yaw_interp)
    )

    def update(i):
        if i >= len(interp_trajectory):
            return

        x, y, z, roll, pitch, yaw = interp_trajectory[i]
        position = np.array([x, y, z])
        R = rotation_matrix_from_euler(roll, pitch, yaw)

        # Update platform position
        tops_world = [R @ p + position for p in tops]
        platform_x = [p[0] for p in tops_world] + [tops_world[0][0]]
        platform_y = [p[1] for p in tops_world] + [tops_world[0][1]]
        platform_z = [p[2] for p in tops_world] + [tops_world[0][2]]
        platform_polygon.set_data(platform_x, platform_y)
        platform_polygon.set_3d_properties(platform_z)  # type: ignore

        # Update legs
        leg_lengths = []
        for j in range(6):
            start_point = bots[j]
            end_point = tops_world[j]
            legs[j].set_data(
                [start_point[0], end_point[0]], [start_point[1], end_point[1]]
            )
            legs[j].set_3d_properties([start_point[2], end_point[2]])  # type: ignore

            # Compute leg length
            leg_vector = end_point - start_point
            leg_length = np.linalg.norm(leg_vector)
            leg_lengths.append(leg_length)

        # Update leg length bars
        for j, bar in enumerate(leg_bars):
            bar.set_height(leg_lengths[j])

            # Color the bar based on how close it is to limits
            if leg_lengths[j] < leg_min or leg_lengths[j] > leg_max:
                bar.set_color("red")
            else:
                bar.set_color("black")

        # Update leg length text
        leg_text.set_text(
            f"Position: ({x:.2f}, {y:.2f}, {z:.2f})\n"
            f"Orientation: ({np.degrees(roll):.2f}°, "
            f"{np.degrees(pitch):.2f}°, "
            f"{np.degrees(yaw):.2f}°)"
        )

        # Update progress bar
        progress = i / (len(interp_trajectory) - 1)
        progress_bar[0].set_width(progress)
        sim_time = i * len(trajectory) / len(interp_trajectory) * dt
        sim_time_tot = (len(trajectory) - 1) * dt
        time_text.set_text(f"Time: {sim_time:.1f}s / {sim_time_tot:.1f}s")

        return (
            platform_polygon,
            *legs,
            *leg_bars,
            leg_text,
            progress_bar[0],
            time_text,
        )

    frame_count = len(interp_trajectory)
    interval = 1e3 / fps  # convert to milliseconds

    anim = FuncAnimation(
        fig,
        update,  # type: ignore
        frames=frame_count,
        interval=interval,
        blit=True,
        # repeat=False,
        repeat=True,
    )

    return anim, fig


if __name__ == "__main__":
    # Example trajectory: list of [x, y, z, roll, pitch, yaw]
    trajectory = [
        [0, 0, 0.5, 0, 0, 0],
        [0, 0, 0.7, 0.1, 0, 0],
        [0, 0, 0.5, 0, 0.1, 0],
        [0, 0, 0.3, 0, 0, 0.1],
        [0, 0, 0.0, 0, 0, 0],
    ]

    anim, fig = visualize_trajectory(trajectory, dt=0.05, sim_rate=5e-2, fps=30)
    fig.show()
