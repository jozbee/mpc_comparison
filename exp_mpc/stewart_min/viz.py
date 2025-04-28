import typing as tp

import numpy as np
import matplotlib as mpl
import matplotlib.figure as mpl_fig
import matplotlib.axes as mpl_ax
import matplotlib.animation as mpl_anim
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import exp_mpc.stewart_min.spec as spec
import exp_mpc.stewart_min.const as const
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
        [point[0] for point in const.bots] + [point[0] for point in const.tops]
    )
    x_min = np.min(x_lims) - 0.5
    x_max = np.max(x_lims) + 0.5
    y_lims = np.array(
        [point[1] for point in const.bots] + [point[1] for point in const.tops]
    )
    y_min = np.min(y_lims) - 0.5
    y_max = np.max(y_lims) + 0.5
    z_lims = np.array(
        [point[2] for point in const.bots] + [point[2] for point in const.tops]
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
    ax_legs.set_ylim(const.leg_min * 0.9, const.leg_max * 1.1)
    ax_legs.axhline(
        y=const.leg_min, color="r", linestyle="-", alpha=0.5, label="Min Length"
    )
    ax_legs.axhline(
        y=const.leg_max, color="r", linestyle="-", alpha=0.5, label="Max Length"
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
    base_x = [const.bots[i][0] for i in range(6)] + [const.bots[0][0]]
    base_y = [const.bots[i][1] for i in range(6)] + [const.bots[0][1]]
    base_z = [const.bots[i][2] for i in range(6)] + [const.bots[0][2]]
    base_polygon.set_data(base_x, base_y)
    base_polygon.set_3d_properties(base_z)  # type: ignore

    # fig.tight_layout()

    # Interpolate trajectory for smoother animation
    num_points = len(trajectory)
    t_values = np.arange(0, const.dt * num_points, const.dt)
    t_interp = np.arange(0, const.dt * num_points, 1.0 / fps * sim_rate)

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
        tops_world = [R @ p + position for p in const.tops]
        platform_x = [p[0] for p in tops_world] + [tops_world[0][0]]
        platform_y = [p[1] for p in tops_world] + [tops_world[0][1]]
        platform_z = [p[2] for p in tops_world] + [tops_world[0][2]]
        platform_polygon.set_data(platform_x, platform_y)
        platform_polygon.set_3d_properties(platform_z)  # type: ignore

        # Update legs
        leg_lengths = []
        for j in range(6):
            start_point = const.bots[j]
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
            if leg_lengths[j] < const.leg_min or leg_lengths[j] > const.leg_max:
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
        sim_time = i * len(trajectory) / len(interp_trajectory) * const.dt
        sim_time_tot = (len(trajectory) - 1) * const.dt
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


def simple_plot(
    axis: mpl_ax.Axes,
    time: np.ndarray,
    data: np.ndarray,
    title: str,
    data_label: str,
    min_limit: tp.Optional[float] = None,
    max_limit: tp.Optional[float] = None,
    reference: tp.Optional[np.ndarray] = None,
):
    axis.set_title(title)
    axis.set_ylabel(data_label)
    axis.set_xlabel("[s]")
    axis.plot(time, data, color="blue")
    if reference is not None:
        axis.plot(time, reference, color="orange", linestyle="--")
        data = np.concatenate([data, reference])
    if min_limit is not None:
        axis.axhline(y=min_limit, linestyle="-", alpha=0.5, color="red")
    if max_limit is not None:
        axis.axhline(y=max_limit, linestyle="-", alpha=0.5, color="red")
    axis.set_ylim(*_get_limits(data))
    axis.set_xlim(time[0], time[-1])
    axis.grid()


def _reference_helper(
    reference: tp.Optional[np.ndarray], index: int
) -> np.ndarray | None:
    """Helper function to get the reference data for a specific axis."""
    if reference is None:
        return None
    if reference.ndim == 1:
        return reference
    if reference.ndim == 2 and reference.shape[1] == 3:
        return reference[:, index]
    raise ValueError(
        "Invalid reference shape. Must be 1D or 2D with shape (N, 3)."
    )


def _plot_cartesian_trajectory(
    axes: np.ndarray,
    trajectory: list[spec.TableSol],
    xyz_fun: tp.Callable,
    angle_fun: tp.Callable,
):
    """Common cartesian trajectory routines, for positions."""
    assert all(type(ax) is mpl_ax.Axes for ax in axes.flatten())
    assert len(trajectory) > 0

    times = np.arange(0, len(trajectory), dtype=float) * const.dt

    ################
    # xyz position #
    ################

    # compute
    xyzs = np.array([xyz_fun(sol) for sol in trajectory])

    # setup
    ax_x = axes[0, 0]
    ax_y = axes[0, 1]
    ax_z = axes[0, 2]

    # plot
    simple_plot(
        axis=ax_x,
        time=times,
        data=xyzs[:, 0],
        title="X Position",
        data_label="[m]",
    )
    simple_plot(
        axis=ax_y,
        time=times,
        data=xyzs[:, 1],
        title="Y Position",
        data_label="[m]",
    )
    simple_plot(
        axis=ax_z,
        time=times,
        data=xyzs[:, 2],
        title="Z Position",
        data_label="[m]",
    )

    ####################
    # angular position #
    ####################

    # compute
    angles = np.array([angle_fun(sol) for sol in trajectory])

    # setup
    ax_roll = axes[1, 0]
    ax_pitch = axes[1, 1]
    ax_yaw = axes[1, 2]

    # plot
    simple_plot(
        axis=ax_roll,
        time=times,
        data=angles[:, 0],
        title="Roll Angle",
        data_label="[rad]",
        min_limit=-const.max_roll,
        max_limit=const.max_roll,
    )
    simple_plot(
        axis=ax_pitch,
        time=times,
        data=angles[:, 1],
        title="Pitch Angle",
        data_label="[rad]",
        min_limit=-const.max_pitch,
        max_limit=const.max_pitch,
    )
    simple_plot(
        axis=ax_yaw,
        time=times,
        data=angles[:, 2],
        title="Yaw Angle",
        data_label="[rad]",
        min_limit=-const.max_yaw,
        max_limit=const.max_yaw,
    )


def _plot_cartesian_trajectory_p(
    axes: np.ndarray,
    trajectory: list[spec.TableSol],
    xyz_vel_fun: tp.Callable,
    angle_vel_fun: tp.Callable,
    xyz_acc_fun: tp.Callable,
    angle_acc_fun: tp.Callable,
    references: dict[str, np.ndarray] = {},
):
    """Common cartesian trajectory routines, for derivatives."""
    assert all(type(ax) is mpl_ax.Axes for ax in axes.flatten())

    times = np.arange(0, len(trajectory), dtype=float) * const.dt

    ################
    # xyz velocity #
    ################

    # compute
    xyz_vels = np.array([xyz_vel_fun(sol) for sol in trajectory])

    # setup
    ax_x_vel = axes[0, 0]
    ax_y_vel = axes[0, 1]
    ax_z_vel = axes[0, 2]

    # plot
    simple_plot(
        axis=ax_x_vel,
        time=times,
        data=xyz_vels[:, 0],
        title="X Velocity",
        data_label="[m/s]",
        min_limit=-const.max_cart_vel,
        max_limit=const.max_cart_vel,
    )
    simple_plot(
        axis=ax_y_vel,
        time=times,
        data=xyz_vels[:, 1],
        title="Y Velocity",
        data_label="[m/s]",
        min_limit=-const.max_cart_vel,
        max_limit=const.max_cart_vel,
    )
    simple_plot(
        axis=ax_z_vel,
        time=times,
        data=xyz_vels[:, 2],
        title="Z Velocity",
        data_label="[m/s]",
        min_limit=-const.max_cart_vel,
        max_limit=const.max_cart_vel,
    )

    ####################
    # angular velocity #
    ####################

    # compute
    angle_vels = np.array([angle_vel_fun(sol) for sol in trajectory])

    # setup
    ax_omega_x_vel = axes[1, 0]
    ax_omega_y_vel = axes[1, 1]
    ax_omega_z_vel = axes[1, 2]

    # references?
    angle_vel_ref = None
    if "angular-velocity" in references:
        angle_vel_ref = references["angular-velocity"]

    # plot
    simple_plot(
        axis=ax_omega_x_vel,
        time=times,
        data=angle_vels[:, 0],
        title="X Angular Velocity",
        data_label="[rad/s]",
        min_limit=-const.max_angle_vel,
        max_limit=const.max_angle_vel,
        reference=_reference_helper(angle_vel_ref, 0),
    )
    simple_plot(
        axis=ax_omega_y_vel,
        time=times,
        data=angle_vels[:, 1],
        title="Y Angular Velocity",
        data_label="[rad/s]",
        min_limit=-const.max_angle_vel,
        max_limit=const.max_angle_vel,
        reference=_reference_helper(angle_vel_ref, 1),
    )
    simple_plot(
        axis=ax_omega_z_vel,
        time=times,
        data=angle_vels[:, 2],
        title="Z Angular Velocity",
        data_label="[rad/s]",
        min_limit=-const.max_angle_vel,
        max_limit=const.max_angle_vel,
        reference=_reference_helper(angle_vel_ref, 2),
    )

    ####################
    # xyz acceleration #
    ####################

    # compute
    xyz_accs = np.array([xyz_acc_fun(sol) for sol in trajectory])

    # setup
    ax_x_acc = axes[2, 0]
    ax_y_acc = axes[2, 1]
    ax_z_acc = axes[2, 2]

    # references?
    xyz_acc_ref = None
    if "xyz-acceleration" in references:
        xyz_acc_ref = references["xyz-acceleration"]

    # plot
    simple_plot(
        axis=ax_x_acc,
        time=times,
        data=xyz_accs[:, 0],
        title="X Acceleration",
        data_label="[m/s^2]",
        min_limit=-const.max_cart_acc,
        max_limit=const.max_cart_acc,
        reference=_reference_helper(xyz_acc_ref, 0),
    )
    simple_plot(
        axis=ax_y_acc,
        time=times,
        data=xyz_accs[:, 1],
        title="Y Acceleration",
        data_label="[m/s^2]",
        min_limit=-const.max_cart_acc,
        max_limit=const.max_cart_acc,
        reference=_reference_helper(xyz_acc_ref, 1),
    )
    simple_plot(
        axis=ax_z_acc,
        time=times,
        data=xyz_accs[:, 2],
        title="Z Acceleration",
        data_label="[m/s^2]",
        min_limit=-const.max_cart_acc,
        max_limit=const.max_cart_acc,
        reference=_reference_helper(xyz_acc_ref, 2),
    )

    ########################
    # angular acceleration #
    ########################

    # compute
    angle_accs = np.array([angle_acc_fun(sol) for sol in trajectory])

    # setup
    ax_omega_x_acc = axes[3, 0]
    ax_omega_y_acc = axes[3, 1]
    ax_omega_z_acc = axes[3, 2]

    # plot
    simple_plot(
        axis=ax_omega_x_acc,
        time=times,
        data=angle_accs[:, 0],
        title="X Angular Acceleration",
        data_label="[rad/s^2]",
        min_limit=-const.max_angle_acc,
        max_limit=const.max_angle_acc,
    )
    simple_plot(
        axis=ax_omega_y_acc,
        time=times,
        data=angle_accs[:, 1],
        title="Y Angular Acceleration",
        data_label="[rad/s^2]",
        min_limit=-const.max_angle_acc,
        max_limit=const.max_angle_acc,
    )
    simple_plot(
        axis=ax_omega_z_acc,
        time=times,
        data=angle_accs[:, 2],
        title="Z Angular Acceleration",
        data_label="[rad/s^2]",
        min_limit=-const.max_angle_acc,
        max_limit=const.max_angle_acc,
    )


def plot_human_trajectory(
    trajectory: list[spec.TableSol],
    references: dict[str, np.ndarray] = {},
    fig_title: str = "Head Trajectory",
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
    fig_title :
        Suptitle of the figure.
    fig_kwds :
        Other figure keywords.

    Returns
    -------
    Figure with xyz-velocity, angular-velocity, xyz-acceleration, and
    angular acceleration.
    """
    # setup
    gridspec_kw = {
        # "height_ratios": [1.0, 1.0, 1.0, 1.0],
        # "width_ratios": [1.0, 1.0, 1.0],
        "wspace": 0.35,
        "hspace": 0.7,
    }
    fig, axes = plt.subplots(
        nrows=4, ncols=3, gridspec_kw=gridspec_kw, figsize=(16, 10), **fig_kwds
    )
    fig.suptitle(fig_title, fontsize=16)
    _plot_cartesian_trajectory_p(
        axes=axes,
        trajectory=trajectory,
        xyz_vel_fun=utils.human_vel,
        angle_vel_fun=utils.human_angle_vel,
        xyz_acc_fun=utils.human_acc,
        angle_acc_fun=utils.human_angle_acc,
        references=references,
    )
    return fig


def plot_cartesian_table_trajectory(
    trajectory: list[spec.TableSol],
    fig_title: str = "Table Trajectory",
    fig_kwds: dict = {},
) -> mpl_fig.Figure:
    """Plot the table trajectory from the solutions of a simulation run.

    Parameters
    ----------
    trajectory :
        List of initial conditions to plot.
    references :
        References that the table should follow.
        Supports keys 'xyz-acceleration' and 'angular-velocity'.
        The values should be arrays with shape (len(trajectory), 3).
    fig_title :
        Suptitle of the figure.
    fig_kwds :
        Other figure keywords.

    Returns
    -------
    Figure with xyz-velocity, angular-velocity, xyz-acceleration, and
    angular acceleration.
    """
    gs_kwds = {
        # "height_ratios": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        # "width_ratios": [1.0, 1.0, 1.0],
        "wspace": 0.35,
        "hspace": 1.0,
    }
    fig, axes = plt.subplots(
        nrows=6, ncols=3, gridspec_kw=gs_kwds, figsize=(16, 10), **fig_kwds
    )
    fig.suptitle(fig_title, fontsize=16)
    _plot_cartesian_trajectory(
        axes=axes[:2, :],
        trajectory=trajectory,
        xyz_fun=utils.table_pos,
        angle_fun=utils.table_angle,
    )
    _plot_cartesian_trajectory_p(
        axes=axes[2:, :],
        trajectory=trajectory,
        xyz_vel_fun=utils.table_vel,
        angle_vel_fun=utils.table_angle_vel,
        xyz_acc_fun=utils.table_acc,
        angle_acc_fun=utils.table_angle_acc,
    )
    return fig


def plot_actuator_trajectory(
    trajectory: list[spec.TableSol],
    fig_title: str = "Actuator Trajectory",
    fig_kwds: dict = {},
) -> mpl_fig.Figure:
    """Plot the actuator trajectory from the solutions of a simulation run.

    Specifically, we plot leg lengths, leg velocities, leg accelerations,
    and joint angles.

    Parameters
    ----------
    trajectory :
        List of initial conditions to plot.
    fig_title :
        Suptitle of the figure.
    fig_kwds :
        Other figure keywords.

    Returns
    -------
    Figure with actuator lengths, velocities, accelerations, and joint angles.
    """
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 10),
        gridspec_kw={"wspace": 0.35},
        **fig_kwds,
    )
    fig.suptitle(fig_title, fontsize=16)

    times = np.arange(0, len(trajectory), dtype=float) * const.dt

    colors = [mpl.colormaps["viridis"](c) for c in np.linspace(0, 1, 6)]

    ###############
    # leg lengths #
    ###############

    leg_pos = [utils.leg_pos(sol) for sol in trajectory]
    leg_pos = np.array(leg_pos)

    ax_pos = axes[0, 0]
    ax_pos.set_title("Leg Lengths")
    ax_pos.set_xlabel("[s]")
    ax_pos.set_ylabel("[m]")
    ax_pos.set_xlim(times[0], times[-1])
    ax_pos.set_ylim(*_get_limits(leg_pos))
    ax_pos.grid()

    for leg_num, color in zip(range(leg_pos.shape[1]), colors):
        ax_pos.plot(
            times,
            leg_pos[:, leg_num],
            color=color,
            label=f"Leg {leg_num + 1}",
        )
    ax_pos.axhline(
        y=const.leg_min,
        linestyle="-",
        alpha=0.5,
        color="red",
        label="Min Length",
    )
    ax_pos.axhline(
        y=const.leg_max,
        linestyle="-",
        alpha=0.5,
        color="red",
        label="Max Length",
    )
    ax_pos.legend()

    ##################
    # leg velocities #
    ##################

    leg_vel = [utils.leg_vel(sol) for sol in trajectory]
    leg_vel = np.array(leg_vel)

    ax_vel = axes[0, 1]
    ax_vel.set_title("Leg Velocities")
    ax_vel.set_xlabel("[s]")
    ax_vel.set_ylabel("[m/s]")
    ax_vel.set_xlim(times[0], times[-1])
    ax_vel.set_ylim(*_get_limits(leg_vel))
    ax_vel.grid()

    for leg_num, color in zip(range(leg_vel.shape[1]), colors):
        ax_vel.plot(
            times,
            leg_vel[:, leg_num],
            color=color,
            label=f"Leg {leg_num + 1}",
        )
    ax_vel.axhline(
        y=-const.max_leg_vel,
        linestyle="-",
        alpha=0.5,
        color="red",
        label="Min Velocity",
    )
    ax_vel.axhline(
        y=const.max_leg_vel,
        linestyle="-",
        alpha=0.5,
        color="red",
        label="Max Velocity",
    )
    ax_vel.legend()

    #####################
    # leg accelerations #
    #####################

    leg_acc = [utils.leg_acc(sol) for sol in trajectory]
    leg_acc = np.array(leg_acc)

    ax_acc = axes[1, 0]
    ax_acc.set_title("Leg Accelerations")
    ax_acc.set_xlabel("[s]")
    ax_acc.set_ylabel("[m/s^2]")
    ax_acc.set_xlim(times[0], times[-1])
    ax_acc.set_ylim(*_get_limits(leg_acc))
    ax_acc.grid()

    for leg_num, color in zip(range(leg_acc.shape[1]), colors):
        ax_acc.plot(
            times,
            leg_acc[:, leg_num],
            color=color,
            label=f"Leg {leg_num + 1}",
        )
    ax_acc.legend()

    ################
    # joint angles #
    ################

    top_joint_angles = [utils.angle_joint_top(sol) for sol in trajectory]
    top_joint_angles = np.array(top_joint_angles)
    top_joint_angles = np.degrees(top_joint_angles)

    bot_joint_angles = [utils.angle_joint_bot(sol) for sol in trajectory]
    bot_joint_angles = np.array(bot_joint_angles)
    bot_joint_angles = np.degrees(bot_joint_angles)

    ax_angles = axes[1, 1]
    ax_angles.set_title("Joint Angles")
    ax_angles.set_xlabel("[s]")
    ax_angles.set_ylabel("[deg]")
    ax_angles.set_xlim(times[0], times[-1])
    y_lim_data = np.concatenate(
        [top_joint_angles.flatten(), bot_joint_angles.flatten()]
    )
    ax_angles.set_ylim(*_get_limits(y_lim_data))
    ax_angles.grid()

    joint_colors = [mpl.colormaps["viridis"](c) for c in np.linspace(0, 1, 12)]

    for leg_num, color in zip(
        range(top_joint_angles.shape[1]), joint_colors[:6]
    ):
        ax_angles.plot(
            times,
            top_joint_angles[:, leg_num],
            color=color,
            label=f"Top Joint {leg_num + 1}",
        )
    for leg_num, color in zip(
        range(bot_joint_angles.shape[1]), joint_colors[6:]
    ):
        ax_angles.plot(
            times,
            bot_joint_angles[:, leg_num],
            color=color,
            label=f"Bottom Joint {leg_num + 1}",
        )
    ax_angles.axhline(
        y=np.degrees(const.joint_max_angle),
        linestyle="-",
        alpha=0.5,
        color="red",
        label="Max Joint Angle",
    )
    ax_angles.legend()

    return fig


def split_tablesol(table_sol: spec.TableSol) -> list[spec.TableSol]:
    """Split a TableSol into a list of of 2 point horizon TableSol."""
    assert table_sol.u.shape[0] > 1
    split_sols = []
    for i in range(table_sol.u.shape[0] - 1):
        split_sols.append(table_sol[i : i + 2])
    return split_sols


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
