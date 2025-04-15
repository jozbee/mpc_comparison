import numpy as np
import matplotlib.figure as mpl_fig
import matplotlib.animation as mpl_anim
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import exp_mpc.stewart_min.spec as spec
import exp_mpc.stewart_min.utils as utils

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def waypoints_from_solutions(
    trajectory: list[spec.TableSol],
) -> list[np.ndarray]:
    """Convert a list of solutions to a list of waypoints."""
    return [np.array(sol.pose_at(0)) for sol in trajectory]


def animate_trajectory(
    trajectory: list[list[float]] | list[np.ndarray] | list[spec.TableSol],
    sim_rate: float = 1.0,
    fps: float = 30.0,
) -> tuple[mpl_anim.FuncAnimation, mpl_fig.Figure]:
    """Visualize the Stewart platform following a trajectory.

    Parameters
    ----------
    trajectory :
        List of waypoints, where each waypoint is a list or array of
        [x, y, z, roll, pitch, yaw] defining the platform's position and
        orientation.
    sim_rate :
        Simulation rate multiplier (1.0 = real-time, 2.0 = twice as fast).
    fps :
        Frames per second for the animation.

    Returns
    -------
    Animation object that can be further processed (e.g., saved to file), and
    its corresponding figure.
    """
    assert type(trajectory) is list
    assert len(trajectory) > 0

    # possible conversion needed
    waypoints: list[np.ndarray] | list[list[float]]
    if type(trajectory[0]) is spec.TableSol:
        assert all(type(sol) is spec.TableSol for sol in trajectory)
        waypoints = waypoints_from_solutions(trajectory)  # type: ignore
    else:
        waypoints = trajectory  # type: ignore

    # Setup figure
    fig = plt.figure(figsize=(10, 6))
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

    # Create a margin around the workspace
    x_lims = np.array(
        [point[0] for point in spec.bots] + [point[0] for point in spec.tops]
    )
    x_min = np.min(x_lims) - 0.5
    x_max = np.max(x_lims) + 0.5
    y_lims = np.array(
        [point[1] for point in spec.bots] + [point[1] for point in spec.tops]
    )
    y_min = np.min(y_lims) - 0.5
    y_max = np.max(y_lims) + 0.5
    z_lims = np.array(
        [point[2] for point in spec.bots] + [point[2] for point in spec.tops]
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
    ax_legs.set_ylim(spec.leg_min * 0.9, spec.leg_max * 1.1)
    ax_legs.axhline(
        y=spec.leg_min, color="r", linestyle="-", alpha=0.5, label="Min Length"
    )
    ax_legs.axhline(
        y=spec.leg_max, color="r", linestyle="-", alpha=0.5, label="Max Length"
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
    base_x = [spec.bots[i][0] for i in range(6)] + [spec.bots[0][0]]
    base_y = [spec.bots[i][1] for i in range(6)] + [spec.bots[0][1]]
    base_z = [spec.bots[i][2] for i in range(6)] + [spec.bots[0][2]]
    base_polygon.set_data(base_x, base_y)
    base_polygon.set_3d_properties(base_z)  # type: ignore

    # fig.tight_layout()

    # Interpolate trajectory for smoother animation
    num_points = len(trajectory)
    t_values = np.arange(0, spec.dt * num_points, spec.dt)
    t_interp = np.arange(0, spec.dt * num_points, 1.0 / fps * sim_rate)

    # Interpolate positions (x, y, z)
    positions = np.array([point[:3] for point in waypoints])
    x_interp = np.interp(t_interp, t_values, positions[:, 0])
    y_interp = np.interp(t_interp, t_values, positions[:, 1])
    z_interp = np.interp(t_interp, t_values, positions[:, 2])

    # Interpolate orientations (roll, pitch, yaw)
    orientations = np.array([point[3:] for point in waypoints])
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
        R = utils._get_R(roll, pitch, yaw)

        # Update platform position
        tops_world = [R @ p + position for p in spec.tops]
        platform_x = [p[0] for p in tops_world] + [tops_world[0][0]]
        platform_y = [p[1] for p in tops_world] + [tops_world[0][1]]
        platform_z = [p[2] for p in tops_world] + [tops_world[0][2]]
        platform_polygon.set_data(platform_x, platform_y)
        platform_polygon.set_3d_properties(platform_z)  # type: ignore

        # Update legs
        leg_lengths = []
        for j in range(6):
            start_point = spec.bots[j]
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
            if leg_lengths[j] < spec.leg_min or leg_lengths[j] > spec.leg_max:
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
        sim_time = i * len(trajectory) / len(interp_trajectory) * spec.dt
        sim_time_tot = (len(trajectory) - 1) * spec.dt
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

    anim = mpl_anim.FuncAnimation(
        fig,
        update,  # type: ignore
        frames=frame_count,
        interval=interval,
        blit=True,
        # repeat=False,
        repeat=True,
    )

    return anim, fig


def _get_limits(data: np.ndarray) -> tuple[float, float]:
    """Get reasonable plotting limits for some data."""
    eps = 2**-4  # magic

    minimum = np.min(data)
    maximum = np.max(data)
    diff = maximum - minimum

    # edge case
    if minimum == 0.0 and maximum == 0.0:
        return (-0.2, 0.2)

    limit_diff = diff * eps
    limits = (
        minimum - limit_diff,
        maximum + limit_diff,
    )
    return limits


def plot_head_trajectory(
    trajectory: list[spec.TableSol],
    references: dict[str, np.ndarray] = {},
    fig_kwds: dict = {},
) -> mpl_fig.Figure:
    """Plot the head trajectory from the solutions of a simulation run.

    Parameters
    ----------
    trajectory :
        List of initial conditions to plot.
    references :
        References that the head should follow.
        Supports keys 'xyz-acceleration' and 'angular-velocity'.
        The values should be arrays with shape (len(trajectory), 3).
    fig_kwds :
        Other figure keywords.

    Returns
    -------
    Figure with xyz-velocity, angular-velocity, xyz-acceleration, and
    angular acceleration.
    """
    # setup
    fig = plt.figure(figsize=(16, 10), **fig_kwds)
    gs = gridspec.GridSpec(
        nrows=4,
        ncols=3,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.35,
        hspace=0.7,
    )

    times = np.arange(0, len(trajectory), dtype=float) * spec.dt

    ################
    # xyz velocity #
    ################

    # compute
    xyz_vels = np.array([utils.human_vel(sol) for sol in trajectory])

    # setup
    ax_x_vel = fig.add_subplot(gs[0, 0])
    ax_y_vel = fig.add_subplot(gs[0, 1])
    ax_z_vel = fig.add_subplot(gs[0, 2])

    ax_x_vel.set_title("X Velocity")
    ax_y_vel.set_title("Y Velocity")
    ax_z_vel.set_title("Z Velocity")

    ax_x_vel.set_ylabel("Velocity (m/s)")
    ax_y_vel.set_ylabel("Velocity (m/s)")
    ax_z_vel.set_ylabel("Velocity (m/s)")

    ax_x_vel.set_xlabel("Time (s)")
    ax_y_vel.set_xlabel("Time (s)")
    ax_z_vel.set_xlabel("Time (s)")

    # plot
    ax_x_vel.plot(times, xyz_vels[:, 0])
    ax_y_vel.plot(times, xyz_vels[:, 1])
    ax_z_vel.plot(times, xyz_vels[:, 2])

    # limits
    ax_x_vel.set_ylim(*_get_limits(xyz_vels[:, 0]))
    ax_y_vel.set_ylim(*_get_limits(xyz_vels[:, 1]))
    ax_z_vel.set_ylim(*_get_limits(xyz_vels[:, 2]))

    ax_x_vel.set_xlim(times[0], times[-1])
    ax_y_vel.set_xlim(times[0], times[-1])
    ax_z_vel.set_xlim(times[0], times[-1])

    # bounds
    ax_x_vel.axhline(
        y=-spec.max_cart_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_x_vel.axhline(
        y=spec.max_cart_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_y_vel.axhline(
        y=-spec.max_cart_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_y_vel.axhline(
        y=spec.max_cart_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_z_vel.axhline(
        y=-spec.max_cart_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_z_vel.axhline(
        y=spec.max_cart_vel,
        linestyle="-",
        alpha=0.5,
    )

    # refine
    ax_x_vel.grid()
    ax_y_vel.grid()
    ax_z_vel.grid()

    ####################
    # angular velocity #
    ####################

    # compute
    angle_vels = np.array([utils.angle_vel(sol) for sol in trajectory])

    # setup
    ax_omega_x_vel = fig.add_subplot(gs[1, 0])
    ax_omega_y_vel = fig.add_subplot(gs[1, 1])
    ax_omega_z_vel = fig.add_subplot(gs[1, 2])

    ax_omega_x_vel.set_title("X Angular Velocity")
    ax_omega_y_vel.set_title("Y Angular Velocity")
    ax_omega_z_vel.set_title("Z Angular Velocity")

    ax_omega_x_vel.set_ylabel("Angular Velocity (rad/s)")
    ax_omega_y_vel.set_ylabel("Angular Velocity (rad/s)")
    ax_omega_z_vel.set_ylabel("Angular Velocity (rad/s)")

    ax_omega_x_vel.set_xlabel("Time (s)")
    ax_omega_y_vel.set_xlabel("Time (s)")
    ax_omega_z_vel.set_xlabel("Time (s)")

    # plot
    ax_omega_x_vel.plot(times, angle_vels[:, 0])
    ax_omega_y_vel.plot(times, angle_vels[:, 1])
    ax_omega_z_vel.plot(times, angle_vels[:, 2])

    # references?
    angle_vel_ref = np.empty(shape=(0, 3))
    if "angular-velocity" in references:
        angle_vel_ref = references["angular-velocity"]
        ax_omega_x_vel.plot(times, angle_vel_ref[:, 0], linestyle="--")
        ax_omega_y_vel.plot(times, angle_vel_ref[:, 1], linestyle="--")
        ax_omega_z_vel.plot(times, angle_vel_ref[:, 2], linestyle="--")

    # limits
    ax_omega_x_vel.set_ylim(
        *_get_limits(np.concatenate([angle_vels[:, 0], angle_vel_ref[:, 0]]))
    )
    ax_omega_y_vel.set_ylim(
        *_get_limits(np.concatenate([angle_vels[:, 1], angle_vel_ref[:, 1]]))
    )
    ax_omega_z_vel.set_ylim(
        *_get_limits(np.concatenate([angle_vels[:, 2], angle_vel_ref[:, 2]]))
    )

    ax_omega_x_vel.set_xlim(times[0], times[-1])
    ax_omega_y_vel.set_xlim(times[0], times[-1])
    ax_omega_z_vel.set_xlim(times[0], times[-1])

    # bounds
    ax_omega_x_vel.axhline(
        y=-spec.max_angle_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_x_vel.axhline(
        y=spec.max_angle_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_y_vel.axhline(
        y=-spec.max_angle_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_y_vel.axhline(
        y=spec.max_angle_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_z_vel.axhline(
        y=-spec.max_angle_vel,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_z_vel.axhline(
        y=spec.max_angle_vel,
        linestyle="-",
        alpha=0.5,
    )

    # refine
    ax_omega_x_vel.grid()
    ax_omega_y_vel.grid()
    ax_omega_z_vel.grid()

    ####################
    # xyz acceleration #
    ####################

    # compute
    xyz_accs = np.array([utils.human_acc(sol) for sol in trajectory])

    # setup
    ax_x_acc = fig.add_subplot(gs[2, 0])
    ax_y_acc = fig.add_subplot(gs[2, 1])
    ax_z_acc = fig.add_subplot(gs[2, 2])

    ax_x_acc.set_title("X Acceleration")
    ax_y_acc.set_title("Y Acceleration")
    ax_z_acc.set_title("Z Acceleration")

    ax_x_acc.set_ylabel("Acceleration (m/s^2)")
    ax_y_acc.set_ylabel("Acceleration (m/s^2)")
    ax_z_acc.set_ylabel("Acceleration (m/s^2)")

    ax_x_acc.set_xlabel("Time (s)")
    ax_y_acc.set_xlabel("Time (s)")
    ax_z_acc.set_xlabel("Time (s)")

    # plot
    ax_x_acc.plot(times, xyz_accs[:, 0])
    ax_y_acc.plot(times, xyz_accs[:, 1])
    ax_z_acc.plot(times, xyz_accs[:, 2])

    # references?
    xyz_acc_ref = np.empty(shape=(0, 3))
    if "xyz-acceleration" in references:
        xyz_acc_ref = references["xyz-acceleration"]
        ax_x_acc.plot(times, xyz_acc_ref[:, 0], linestyle="--")
        ax_y_acc.plot(times, xyz_acc_ref[:, 1], linestyle="--")
        ax_z_acc.plot(times, xyz_acc_ref[:, 2], linestyle="--")

    # limits
    ax_x_acc.set_ylim(
        *_get_limits(np.concatenate([xyz_accs[:, 0], xyz_acc_ref[:, 0]]))
    )
    ax_y_acc.set_ylim(
        *_get_limits(np.concatenate([xyz_accs[:, 1], xyz_acc_ref[:, 1]]))
    )
    ax_z_acc.set_ylim(
        *_get_limits(np.concatenate([xyz_accs[:, 2], xyz_acc_ref[:, 2]]))
    )

    ax_x_acc.set_xlim(times[0], times[-1])
    ax_y_acc.set_xlim(times[0], times[-1])
    ax_z_acc.set_xlim(times[0], times[-1])

    # bounds
    ax_x_acc.axhline(
        y=-spec.max_cart_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_x_acc.axhline(
        y=spec.max_cart_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_y_acc.axhline(
        y=-spec.max_cart_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_y_acc.axhline(
        y=spec.max_cart_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_z_acc.axhline(
        y=-spec.max_cart_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_z_acc.axhline(
        y=spec.max_cart_acc,
        linestyle="-",
        alpha=0.5,
    )

    # refine
    ax_x_acc.grid()
    ax_y_acc.grid()
    ax_z_acc.grid()

    ########################
    # angular acceleration #
    ########################

    # compute
    angle_accs = np.array([utils.angle_acc(sol) for sol in trajectory])

    # setup
    ax_omega_x_acc = fig.add_subplot(gs[3, 0])
    ax_omega_y_acc = fig.add_subplot(gs[3, 1])
    ax_omega_z_acc = fig.add_subplot(gs[3, 2])

    ax_omega_x_acc.set_title("X Angular Acceleration")
    ax_omega_y_acc.set_title("Y Angular Acceleration")
    ax_omega_z_acc.set_title("Z Angular Acceleration")

    ax_omega_x_acc.set_ylabel("Angular Acceleration (rad/s^2)")
    ax_omega_y_acc.set_ylabel("Angular Acceleration (rad/s^2)")
    ax_omega_z_acc.set_ylabel("Angular Acceleration (rad/s^2)")

    ax_omega_x_acc.set_xlabel("Time (s)")
    ax_omega_y_acc.set_xlabel("Time (s)")
    ax_omega_z_acc.set_xlabel("Time (s)")

    # plot
    ax_omega_x_acc.plot(times, angle_accs[:, 0])
    ax_omega_y_acc.plot(times, angle_accs[:, 1])
    ax_omega_z_acc.plot(times, angle_accs[:, 2])

    # limits
    ax_omega_x_acc.set_ylim(*_get_limits(angle_accs[:, 0]))
    ax_omega_y_acc.set_ylim(*_get_limits(angle_accs[:, 1]))
    ax_omega_z_acc.set_ylim(*_get_limits(angle_accs[:, 2]))

    ax_omega_x_acc.set_xlim(times[0], times[-1])
    ax_omega_y_acc.set_xlim(times[0], times[-1])
    ax_omega_z_acc.set_xlim(times[0], times[-1])

    # bounds
    ax_omega_x_acc.axhline(
        y=-spec.max_angle_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_x_acc.axhline(
        y=spec.max_angle_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_y_acc.axhline(
        y=-spec.max_angle_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_y_acc.axhline(
        y=spec.max_angle_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_z_acc.axhline(
        y=-spec.max_angle_acc,
        linestyle="-",
        alpha=0.5,
    )
    ax_omega_z_acc.axhline(
        y=spec.max_angle_acc,
        linestyle="-",
        alpha=0.5,
    )

    # refine
    ax_omega_x_acc.grid()
    ax_omega_y_acc.grid()
    ax_omega_z_acc.grid()

    return fig


if __name__ == "__main__":
    # Example trajectory: list of [x, y, z, roll, pitch, yaw]
    trajectory = [
        [0, 0, 0.5, 0, 0, 0],
        [0, 0, 0.7, 0.1, 0, 0],
        [0, 0, 0.5, 0, 0.1, 0],
        [0, 0, 0.3, 0, 0, 0.1],
        [0, 0, 0.0, 0, 0, 0],
    ]

    anim, fig = animate_trajectory(trajectory, sim_rate=5e-2, fps=30)
    fig.show()
