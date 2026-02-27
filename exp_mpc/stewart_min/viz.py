import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib as mpl
import matplotlib.figure as mpl_fig
import matplotlib.axes as mpl_ax
import matplotlib.animation as mpl_anim
import matplotlib.lines as mpl_lines
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import exp_mpc.stewart_min.const as const
import exp_mpc.stewart_min.comp as comp
import exp_mpc.stewart_min.utils as utils
import exp_mpc.stewart_min.opt as opt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def waypoints_from_solutions(
    trajectory: list[utils.TableSol],
) -> list[np.ndarray]:
    """Convert a list of solutions to a list of waypoints."""
    waypoints = []
    for traj in trajectory:
        s = traj.x.get0()
        pose = np.array([s.x, s.y, s.z, s.roll, s.pitch, s.yaw])
        waypoints.append(pose)
    return waypoints


def animate_trajectory(
    trajectory: list[utils.TableSol],
    sim_rate: float = 1.0,
    fps: float = 30.0,
    frame_range: tp.Optional[tuple[int, int]] = None,
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
    frame_range :
        Only process the given frames, if given.

    Returns
    -------
    Animation object that can be further processed (e.g., saved to file), and
    its corresponding figure.
    """
    assert type(trajectory) is list
    assert len(trajectory) > 0

    # possible conversion needed
    waypoints: list[np.ndarray]
    if type(trajectory[0]) is utils.TableSol:
        assert all(type(sol) is utils.TableSol for sol in trajectory)
        waypoints = waypoints_from_solutions(trajectory)  # type: ignore
    else:
        # The rest of the function expects a trajectory of TableSol
        # to display control inputs.
        raise ValueError(
            "`animate_trajectory` now requires a list of `TableSol`."
        )

    # Setup figure
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(
        7,
        3,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6],
        width_ratios=[0.5, 1.0, 0.7],
        wspace=0.4,
        hspace=0.9,
    )

    # Control plots on the left
    control_axes = []
    ax1 = None
    for i in range(6):
        if i == 0:
            ax = fig.add_subplot(gs[i, 0])
            ax1 = ax
        else:
            ax = fig.add_subplot(gs[i, 0], sharex=ax1, sharey=ax1)
        control_axes.append(ax)
    control_lines = [ax.plot([], [], "b-")[0] for ax in control_axes]
    for i, ax in enumerate(control_axes):
        ax.set_title(f"u[{i}]")
        ax.grid(True)
        if i < 5:
            plt.setp(ax.get_xticklabels(), visible=False)

    control_axes[-1].set_xlabel("Horizon")
    control_abs = np.abs(
        np.concatenate([np.ravel(traj.u.control) for traj in trajectory])
    )
    control_lim = np.mean(control_abs) + np.std(control_abs) * 4.0
    # control_lim = np.max(control_abs)
    control_axes[-1].set_ylim(-control_lim, control_lim)
    control_axes[-1].set_xlim(0, trajectory[0].u.size)
    fig.text(0.04, 0.5, "Control", va="center", rotation="vertical")

    # 3D plot for platform visualization
    ax_3d = fig.add_subplot(gs[:-1, 1], projection="3d")
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
    ax_time = fig.add_subplot(gs[-1, 1:])
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
    ax_legs = fig.add_subplot(gs[:-1, 2])
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
    axis_length = 0.2
    head_axes = [
        ax_3d.plot([], [], [], "r-", linewidth=2)[0],
        ax_3d.plot([], [], [], "g-", linewidth=2)[0],
        ax_3d.plot([], [], [], "b-", linewidth=2)[0],
    ]
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

        # Update control plots
        sol_index = min(
            int(i * num_points / len(interp_trajectory)), num_points - 1
        )
        current_sol: utils.TableSol = trajectory[sol_index]  # type: ignore
        u_horizon = current_sol.u
        horizon_t = np.arange(u_horizon.size)
        for j in range(6):
            control_lines[j].set_data(horizon_t, u_horizon.control[:, j])

        x, y, z, roll, pitch, yaw = interp_trajectory[i]
        position = np.array([x, y, z])
        R = comp.rot(roll, pitch, yaw)
        R_head = np.array(comp.rot(roll, pitch, yaw, use_xy=False))

        # Update platform position
        delta = const.human_displacement
        tops_world = [R @ (p - delta) + delta + position for p in const.tops]
        platform_x = [p[0] for p in tops_world] + [tops_world[0][0]]
        platform_y = [p[1] for p in tops_world] + [tops_world[0][1]]
        platform_z = [p[2] for p in tops_world] + [tops_world[0][2]]
        platform_polygon.set_data(platform_x, platform_y)
        platform_polygon.set_3d_properties(platform_z)  # type: ignore

        # Update head reference frame
        head_pos = const.human_displacement + position
        axis_dirs = R_head @ np.eye(3)
        for axis_line, axis_dir in zip(head_axes, axis_dirs.T):
            end_pos = head_pos + axis_length * axis_dir
            axis_line.set_data(
                [head_pos[0], end_pos[0]], [head_pos[1], end_pos[1]]
            )
            axis_line.set_3d_properties([head_pos[2], end_pos[2]])  # type: ignore

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
            *head_axes,
            *leg_bars,
            leg_text,
            progress_bar[0],
            time_text,
            *control_lines,
        )

    frame_count = len(interp_trajectory)
    interval = 1e3 / fps  # convert to milliseconds

    anim_frames = frame_count
    if frame_range is not None:
        anim_frames = range(frame_range[0], frame_range[1])

    anim = mpl_anim.FuncAnimation(
        fig,
        update,  # type: ignore
        frames=anim_frames,
        interval=interval,
        blit=True,
        repeat=False,
        # repeat=True,
    )

    return anim, fig


def animate_human_trajectory(
    trajectory: list[utils.TableSol],
    sim_rate: float = 1.0,
    fps: float = 30.0,
    references: dict[str, np.ndarray] = {},
    frame_range: tp.Optional[tuple[int, int]] = None,
) -> tuple[mpl_anim.FuncAnimation, mpl_fig.Figure]:
    """Animate the human trajectory from the solutions of a simulation run.

    Each frame of the animation shows the MPC horizon for various human
    trajectory variables.

    Parameters
    ----------
    trajectory :
        List of TableSol objects, where each contains the MPC solution at a time step.
    sim_rate :
        Simulation rate multiplier.
    fps :
        Frames per second for the animation.
    references :
        References that the head should follow.
        Supports keys 'xyz-acceleration' and 'angular-velocity'.
        The values should be arrays with shape (len(trajectory), 3).
    frame_range :
        Only process the given frames, if given.

    Returns
    -------
    A tuple containing the animation object and its corresponding figure.
    """
    assert type(trajectory) is list and len(trajectory) > 0
    assert all(isinstance(sol, utils.TableSol) for sol in trajectory)

    # Setup figure and axes, similar to plot_human_trajectory
    gridspec_kw = {
        "wspace": 0.35,
        "hspace": 0.7,
    }
    fig, axes = plt.subplots(
        nrows=4, ncols=3, gridspec_kw=gridspec_kw, figsize=(16, 10)
    )
    fig.suptitle("Human Trajectory Animation", fontsize=16)

    # Create line objects for each plot
    lines = np.array([[ax.plot([], [])[0] for ax in row] for row in axes])
    ref_lines = np.array(
        [
            [ax.plot([], [], linestyle="--", color="orange")[0] for ax in row]
            for row in axes
        ]
    )

    def setup_plot(ax, title, ylabel, min_limit=None, max_limit=None):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Horizon")
        ax.grid(True)
        if min_limit is not None:
            ax.axhline(y=min_limit, linestyle="-", alpha=0.5, color="red")
        if max_limit is not None:
            ax.axhline(y=max_limit, linestyle="-", alpha=0.5, color="red")

    # Setup for each plot
    plot_setups = [
        # Row 0: xyz velocity
        (
            axes[0, 0],
            "X Velocity",
            "[m/s]",
            -const.max_cart_vel,
            const.max_cart_vel,
        ),
        (
            axes[0, 1],
            "Y Velocity",
            "[m/s]",
            -const.max_cart_vel,
            const.max_cart_vel,
        ),
        (
            axes[0, 2],
            "Z Velocity",
            "[m/s]",
            -const.max_cart_vel,
            const.max_cart_vel,
        ),
        # Row 1: angular velocity
        (
            axes[1, 0],
            "X Angular Velocity",
            "[rad/s]",
            -const.max_angle_vel,
            const.max_angle_vel,
        ),
        (
            axes[1, 1],
            "Y Angular Velocity",
            "[rad/s]",
            -const.max_angle_vel,
            const.max_angle_vel,
        ),
        (
            axes[1, 2],
            "Z Angular Velocity",
            "[rad/s]",
            -const.max_angle_vel,
            const.max_angle_vel,
        ),
        # Row 2: xyz acceleration
        (
            axes[2, 0],
            "X Acceleration",
            "[m/s^2]",
            -const.max_cart_acc,
            const.max_cart_acc,
        ),
        (
            axes[2, 1],
            "Y Acceleration",
            "[m/s^2]",
            -const.max_cart_acc,
            const.max_cart_acc,
        ),
        (
            axes[2, 2],
            "Z Acceleration",
            "[m/s^2]",
            -const.max_cart_acc,
            const.max_cart_acc,
        ),
        # Row 3: angular acceleration
        (
            axes[3, 0],
            "X Angular Acceleration",
            "[rad/s^2]",
            -const.max_angle_acc,
            const.max_angle_acc,
        ),
        (
            axes[3, 1],
            "Y Angular Acceleration",
            "[rad/s^2]",
            -const.max_angle_acc,
            const.max_angle_acc,
        ),
        (
            axes[3, 2],
            "Z Angular Acceleration",
            "[rad/s^2]",
            -const.max_angle_acc,
            const.max_angle_acc,
        ),
    ]

    for ax, title, ylabel, min_lim, max_lim in plot_setups:
        setup_plot(ax, title, ylabel, min_lim, max_lim)

    # Set initial axis limits
    horizon_len = trajectory[0].x.size
    for row in axes:
        for ax in row:
            ax.set_xlim(0, horizon_len - 1)

    # Determine global y-limits for each row to keep them constant during animation
    all_vels, all_ang_vels, all_accs, all_ang_accs = [], [], [], []
    for sol in trajectory:
        all_vels.append(utils.human_vel_horizon(sol))
        all_ang_vels.append(utils.human_angle_vel_horizon(sol))
        all_accs.append(utils.human_acc_horizon(sol))
        all_ang_accs.append(utils.human_angle_acc_horizon(sol))

    ylim_data = [
        np.array(all_vels),
        np.array(all_ang_vels),
        np.array(all_accs),
        np.array(all_ang_accs),
    ]
    for i, row_axes in enumerate(axes):
        for j, ax in enumerate(row_axes):
            min_val = np.min(ylim_data[i][:, :, j])
            max_val = np.max(ylim_data[i][:, :, j])
            ax.set_ylim(*_get_limits(jnp.array([min_val, max_val])))

    num_frames = int(len(trajectory) * const.dt * fps / sim_rate)

    def update(frame_num):
        # Map frame number to trajectory index
        traj_index = int(frame_num * (len(trajectory) / num_frames))
        traj_index = min(traj_index, len(trajectory) - 1)

        current_sol = trajectory[traj_index]
        horizon_len = current_sol.u.size
        horizon_t = np.arange(horizon_len)

        # Extract horizon data
        vel_h = utils.human_vel_horizon(current_sol)
        ang_vel_h = utils.human_angle_vel_horizon(current_sol)
        acc_h = utils.human_acc_horizon(current_sol)
        ang_acc_h = utils.human_angle_acc_horizon(current_sol)

        data_horizons = [vel_h, ang_vel_h, acc_h, ang_acc_h]

        # Update plot lines
        for i in range(4):  # rows
            for j in range(3):  # cols
                lines[i, j].set_data(horizon_t, data_horizons[i][:, j])

        # Update reference lines
        ref_map = {
            "angular-velocity": 1,  # row index
            "xyz-acceleration": 2,  # row index
        }
        for ref_key, row_idx in ref_map.items():
            if ref_key in references:
                ref_data = references[ref_key]
                for j in range(3):  # col index
                    if ref_data.ndim == 1:
                        ref_val = ref_data[traj_index]
                    elif ref_data.ndim == 2 and ref_data.shape[1] == 3:
                        ref_val = ref_data[traj_index, j]
                    else:
                        raise RuntimeError(
                            f"ref_data.shape is bad {ref_data.shape}"
                        )

                    ref_lines[row_idx, j].set_data(
                        horizon_t, np.full(horizon_len, ref_val)
                    )

        return [line for row in lines for line in row] + [
            line for row in ref_lines for line in row
        ]

    anim_num_frames = num_frames
    if frame_range is not None:
        anim_num_frames = range(frame_range[0], frame_range[1])
    anim = mpl_anim.FuncAnimation(
        fig,
        update,
        frames=anim_num_frames,
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )

    return anim, fig


def _get_limits(data: jax.Array) -> tuple[float, float]:
    """Get reasonable plotting limits for some data."""
    eps = 2**-4  # magic

    minimum = jnp.min(data)
    maximum = jnp.max(data)
    diff = maximum - minimum

    # edge case
    if minimum == 0.0 and maximum == 0.0:
        return (-0.2, 0.2)

    limit_diff = diff * eps
    limits = (
        float(minimum - limit_diff),
        float(maximum + limit_diff),
    )
    return limits


def simple_plot(
    axis: mpl_ax.Axes,
    time: jax.Array,
    data: jax.Array,
    title: str,
    data_label: str,
    min_limit: tp.Optional[float] = None,
    max_limit: tp.Optional[float] = None,
    reference: tp.Optional[jax.Array] = None,
):
    axis.set_title(title)
    axis.set_ylabel(data_label)
    axis.set_xlabel("[s]")
    axis.plot(time, data, color="blue")
    if reference is not None:
        axis.plot(time, reference, color="orange", linestyle="--")
        data = jnp.concatenate([data, reference])
    if min_limit is not None:
        axis.axhline(y=min_limit, linestyle="-", alpha=0.5, color="red")
    if max_limit is not None:
        axis.axhline(y=max_limit, linestyle="-", alpha=0.5, color="red")
    axis.set_ylim(*_get_limits(data))
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.grid()


def _reference_helper(
    reference: tp.Optional[jax.Array], index: int
) -> jax.Array | None:
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
    trajectory: list[utils.TableSol],
    xyz_fun: tp.Callable,
    angle_fun: tp.Callable,
):
    """Common cartesian trajectory routines, for positions."""
    assert all(type(ax) is mpl_ax.Axes for ax in axes.flatten())
    assert len(trajectory) > 0

    times = jnp.arange(0, len(trajectory), dtype=float) * const.dt

    ################
    # xyz position #
    ################

    # compute
    xyzs = jnp.array([xyz_fun(sol) for sol in trajectory])

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
    angles = jnp.array([angle_fun(sol) for sol in trajectory])

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
        data_label="[deg]",
        min_limit=-np.degrees(const.max_roll),
        max_limit=np.degrees(const.max_roll),
    )
    simple_plot(
        axis=ax_pitch,
        time=times,
        data=angles[:, 1],
        title="Pitch Angle",
        data_label="[deg]",
        min_limit=-np.degrees(const.max_pitch),
        max_limit=np.degrees(const.max_pitch),
    )
    simple_plot(
        axis=ax_yaw,
        time=times,
        data=angles[:, 2],
        title="Yaw Angle",
        data_label="[deg]",
        min_limit=-np.degrees(const.max_yaw),
        max_limit=np.degrees(const.max_yaw),
    )


def _plot_cartesian_trajectory_p(
    axes: np.ndarray,
    trajectory: list[utils.TableSol],
    xyz_vel_fun: tp.Callable,
    angle_vel_fun: tp.Callable,
    xyz_acc_fun: tp.Callable,
    angle_acc_fun: tp.Callable,
    references: dict[str, jax.Array] = {},
):
    """Common cartesian trajectory routines, for derivatives."""
    assert all(type(ax) is mpl_ax.Axes for ax in axes.flatten())

    times = jnp.arange(0, len(trajectory), dtype=float) * const.dt

    ################
    # xyz velocity #
    ################

    # compute
    xyz_vels = jnp.array([xyz_vel_fun(sol) for sol in trajectory])

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
    angle_vels = jnp.array([angle_vel_fun(sol) for sol in trajectory])

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
    xyz_accs = jnp.array([xyz_acc_fun(sol) for sol in trajectory])

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
    angle_accs = jnp.array([angle_acc_fun(sol) for sol in trajectory])

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
    trajectory: list[utils.TableSol],
    references: dict[str, jax.Array] = {},
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


def plot_vestibular_trajectory(
    trajectory: list[utils.TableSol],
    fig_title: str = "Vestibular Trajectory",
    fig_kwds: dict = {},
) -> mpl_fig.Figure:
    #########
    # setup #
    #########

    gridspec_kw = {
        # "height_ratios": [1.0, 1.0, 1.0, 1.0],
        # "width_ratios": [1.0, 1.0, 1.0],
        "wspace": 0.35,
        "hspace": 0.35,
    }
    fig, axes = plt.subplots(
        nrows=2, ncols=3, gridspec_kw=gridspec_kw, figsize=(16, 8), **fig_kwds
    )
    fig.suptitle(fig_title, fontsize=16)

    times = jnp.arange(0, len(trajectory), dtype=float) * const.dt

    ####################
    # angular velocity #
    ####################

    # compute
    @jax.jit
    def get_omegas(sol: utils.TableSol) -> tuple[jax.Array, jax.Array]:
        irl0 = sol.vstate_irl.get0()
        vstate0_irl = jnp.array([irl0.y_omegax, irl0.y_omegay, irl0.y_omegaz])
        sim0 = sol.vstate_sim.get0()
        vstate0_sim = jnp.array([sim0.y_omegax, sim0.y_omegay, sim0.y_omegaz])
        return vstate0_irl, vstate0_sim

    omegas = jnp.array([get_omegas(sol)[0] for sol in trajectory])
    omegas_ref = jnp.array([get_omegas(sol)[1] for sol in trajectory])

    # plot
    simple_plot(
        axis=axes[0, 0],
        time=times,
        data=omegas[:, 0],
        title="X Angular Velocity",
        data_label="[rad/s]",
        min_limit=-const.max_angle_vel,
        max_limit=const.max_angle_vel,
        reference=omegas_ref[:, 0],
    )
    simple_plot(
        axis=axes[0, 1],
        time=times,
        data=omegas[:, 1],
        title="Y Angular Velocity",
        data_label="[rad/s]",
        min_limit=-const.max_angle_vel,
        max_limit=const.max_angle_vel,
        reference=omegas_ref[:, 1],
    )
    simple_plot(
        axis=axes[0, 2],
        time=times,
        data=omegas[:, 2],
        title="Z Angular Velocity",
        data_label="[rad/s]",
        min_limit=-const.max_angle_vel,
        max_limit=const.max_angle_vel,
        reference=omegas_ref[:, 2],
    )

    #######################
    # linear acceleration #
    #######################

    # compute
    @jax.jit
    def get_accs(sol: utils.TableSol) -> tuple[jax.Array, jax.Array]:
        irl0 = sol.vstate_irl.get0()
        vstate0_irl = jnp.array([irl0.y_accx, irl0.y_accy, irl0.y_accz])
        sim0 = sol.vstate_sim.get0()
        vstate0_sim = jnp.array([sim0.y_accx, sim0.y_accy, sim0.y_accz])
        return vstate0_irl, vstate0_sim

    accs = jnp.array([get_accs(sol)[0] for sol in trajectory])
    accs_ref = jnp.array([get_accs(sol)[1] for sol in trajectory])

    # plot
    simple_plot(
        axis=axes[1, 0],
        time=times,
        data=accs[:, 0],
        title="X Linear Acceleration",
        data_label="[m/s^2]",
        min_limit=-const.max_cart_acc,
        max_limit=const.max_cart_acc,
        reference=accs_ref[:, 0],
    )
    simple_plot(
        axis=axes[1, 1],
        time=times,
        data=accs[:, 1],
        title="Y Linear Acceleration",
        data_label="[m/s^2]",
        min_limit=-const.max_cart_acc,
        max_limit=const.max_cart_acc,
        reference=accs_ref[:, 1],
    )
    simple_plot(
        axis=axes[1, 2],
        time=times,
        data=accs[:, 2],
        title="Z Linear Acceleration",
        data_label="[m/s^2]",
        min_limit=-const.max_cart_acc,
        max_limit=const.max_cart_acc,
        reference=accs_ref[:, 2],
    )

    return fig


def plot_cartesian_table_trajectory(
    trajectory: list[utils.TableSol],
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
    trajectory: list[utils.TableSol],
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

    x0s = [sol.x.get0() for sol in trajectory]
    u0s = [sol.u.get0() for sol in trajectory]

    ###############
    # leg lengths #
    ###############

    leg_pos = [utils.leg_pos(x0) for x0 in x0s]
    leg_pos = jnp.array(leg_pos)

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

    leg_vel = [utils.leg_vel(x0) for x0 in x0s]
    leg_vel = jnp.array(leg_vel)

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

    leg_acc = [utils.leg_acc(x0, u0) for x0, u0 in zip(x0s, u0s)]
    leg_acc = jnp.array(leg_acc)

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

    top_joint_angles = [utils.angle_joint_top(x0) for x0 in x0s]
    top_joint_angles = np.array(top_joint_angles)
    top_joint_angles = np.degrees(top_joint_angles)

    bot_joint_angles = [utils.angle_joint_bot(x0) for x0 in x0s]
    bot_joint_angles = np.array(bot_joint_angles)
    bot_joint_angles = np.degrees(bot_joint_angles)

    ax_angles = axes[1, 1]
    ax_angles.set_title("Joint Angles")
    ax_angles.set_xlabel("[s]")
    ax_angles.set_ylabel("[deg]")
    ax_angles.set_xlim(times[0], times[-1])
    y_lim_data = jnp.concatenate(
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


def plot_cost_trajectory(
    acc_refs: jax.Array,
    omega_refs: jax.Array,
    weights: opt.Weights,
    cost_terms: opt.CostTerms,
    trajectory: list[utils.TableSol],
    fig_title: str = "Cost Trajectory",
    fig_kwds: dict = {},
) -> mpl_fig.Figure:
    """Plot the components of the MPC cost function.

    Specifically, we plot the costs corresponding to
    * head acceleration
    * head angular velocity
    * leg position boundary
    * leg velocity boundary
    * joint angle boundary
    * yaw boundary
    * control

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
    Figure with cost terms.
    """
    #########
    # setup #
    #########

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(16, 12),
        gridspec_kw={"wspace": 0.35, "hspace": 0.5},
        **fig_kwds,
    )
    fig.suptitle(fig_title, fontsize=16)
    times = np.arange(0, len(trajectory), dtype=float) * const.dt
    legend_kwargs = dict(
        bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.0
    )

    def colors(num: int) -> list[tuple[float, float, float, float]]:
        return [mpl.colormaps["viridis"](c) for c in np.linspace(0, 1, num)]

    id_weights = opt.Weights()
    scaled_weights = opt.Weights(
        acc=weights.acc,
        omega=weights.omega,
        leg=weights.leg,
        leg_vel=weights.leg_vel,
        joint_angle=weights.joint_angle,
        yaw=weights.yaw,
        control=weights.control,
    )
    full_weights = weights
    assert len(acc_refs.shape) == 3
    assert acc_refs.shape[-1] == 3
    assert len(omega_refs.shape) == 3
    assert omega_refs.shape[-1] == 3

    ############
    # head acc #
    ############

    @jax.jit
    def head_fun(weights: opt.Weights, sol: utils.TableSol) -> jax.Array:
        return 0.5 * jnp.mean(
            opt.acc_cost_arr(
                weights=weights,
                vstate_irl=sol.vstate_irl.pop0(),
                vstate_sim=sol.vstate_sim.pop0(),
                control=sol.u,
            )
        )

    def weight2head_acc(weights: opt.Weights) -> jax.Array:
        vals = [head_fun(weights, sol) for sol in trajectory]
        return jnp.array(vals)

    id_head_acc = weight2head_acc(id_weights)
    scaled_head_acc = weight2head_acc(scaled_weights)
    full_head_acc = weight2head_acc(full_weights)
    all_head_acc = [id_head_acc, scaled_head_acc, full_head_acc]

    ax_head: mpl_ax.Axes = axes[0, 0]
    ax_head.set_title("Head Accelerations")
    ax_head.set_xlabel("[s]")
    ax_head.set_ylabel("[cost]")
    ax_head.set_xlim(times[0], times[-1])
    all_head_vals = jnp.concatenate(all_head_acc)
    ax_head.set_ylim(*_get_limits(all_head_vals))
    ax_head.grid()

    labels = ["id", "scaled", "full"]
    for data, label, color in zip(all_head_acc, labels, colors(3)):
        ax_head.plot(times, data, color=color, label=label)
    ax_head.legend(**legend_kwargs)

    ##############
    # head omega #
    ##############

    @jax.jit
    def omega_fun(weights: opt.Weights, sol: utils.TableSol) -> jax.Array:
        return 0.5 * jnp.mean(
            opt.omega_cost_arr(
                weights=weights,
                vstate_irl=sol.vstate_irl.pop0(),
                vstate_sim=sol.vstate_sim.pop0(),
                control=sol.u,
            )
        )

    def weight2omega(weights: opt.Weights) -> jax.Array:
        vals = [
            omega_fun(weights, sol, ref)
            for sol, ref in zip(trajectory, omega_refs)
        ]
        return jnp.array(vals)

    id_omega = weight2omega(id_weights)
    scaled_omega = weight2omega(scaled_weights)
    full_omega = weight2omega(full_weights)
    all_omega = [id_omega, scaled_omega, full_omega]

    ax_omega: mpl_ax.Axes = axes[0, 1]
    ax_omega.set_title("Head Angular Velocity")
    ax_omega.set_xlabel("[s]")
    ax_omega.set_ylabel("[cost]")
    ax_omega.set_xlim(times[0], times[-1])
    all_omega_vals = jnp.concatenate(all_omega)
    ax_omega.set_ylim(*_get_limits(all_omega_vals))
    ax_omega.grid()

    labels = ["id", "scaled", "full"]
    for data, label, color in zip(all_omega, labels, colors(3)):
        ax_omega.plot(times, data, color=color, label=label)
    ax_omega.legend(**legend_kwargs)

    ##############
    # leg common #
    ##############

    @jax.jit
    def leg_vel_fun(sol: utils.TableSol) -> tuple[jax.Array, jax.Array]:
        length_cost_arr, vel_cost_arr = opt.leg_boundary_cost_arr(
            weights=weights,
            length_cost=cost_terms.leg_cost,
            vel_cost=cost_terms.leg_vel_cost,
            state=sol.x,
            control=sol.u,
        )
        length_cost_val = jnp.mean(length_cost_arr, axis=0)
        vel_cost_val = jnp.mean(vel_cost_arr, axis=0)
        return length_cost_val, vel_cost_val

    leg_pos = []
    leg_vel = []
    for sol in trajectory:
        res = leg_vel_fun(sol)
        leg_pos.append(res[0])
        leg_vel.append(res[1])
    leg_pos = jnp.stack(leg_pos)
    leg_vel = jnp.stack(leg_vel)

    ###########
    # leg pos #
    ###########

    ax_leg_pos: mpl_ax.Axes = axes[1, 0]
    ax_leg_pos.set_title("Leg position boundary")
    ax_leg_pos.set_xlabel("[s]")
    ax_leg_pos.set_ylabel("[cost]")
    ax_leg_pos.set_xlim(times[0], times[-1])
    ax_leg_pos.set_ylim(*_get_limits(jnp.ravel(leg_pos)))
    ax_leg_pos.grid()

    for i, color in enumerate(colors(6)):
        ax_leg_pos.plot(times, leg_pos[:, i], color=color, label=f"leg {i}")
    ax_leg_pos.legend(**legend_kwargs)

    ###########
    # leg vel #
    ###########

    ax_leg_vel: mpl_ax.Axes = axes[1, 1]
    ax_leg_vel.set_title("Leg velocity boundary")
    ax_leg_vel.set_xlabel("[s]")
    ax_leg_vel.set_ylabel("[cost]")
    ax_leg_vel.set_xlim(times[0], times[-1])
    ax_leg_vel.set_ylim(*_get_limits(jnp.ravel(leg_vel)))
    ax_leg_vel.grid()

    for i, color in enumerate(colors(6)):
        ax_leg_vel.plot(times, leg_vel[:, i], color=color, label=f"leg {i}")
    ax_leg_vel.legend(**legend_kwargs)

    #############
    # leg angle #
    #############

    @jax.jit
    def joint_angle_fun(sol: utils.TableSol) -> jax.Array:
        angle_cost_arr = opt.joint_angle_boundary_cost_arr(
            weights=weights,
            cost=cost_terms.joint_angle_cost,
            state=sol.x,
            control=sol.u,
        )
        angle_cost_val = jnp.mean(angle_cost_arr, axis=0)
        return angle_cost_val

    joint_angle = jnp.array([joint_angle_fun(sol) for sol in trajectory])

    ax_leg_angle: mpl_ax.Axes = axes[2, 0]
    ax_leg_angle.set_title("Joint angle boundary")
    ax_leg_angle.set_xlabel("[s]")
    ax_leg_angle.set_ylabel("[cost]")
    ax_leg_angle.set_xlim(times[0], times[-1])
    ax_leg_angle.set_ylim(*_get_limits(jnp.ravel(joint_angle)))
    ax_leg_angle.grid()

    for i, color in enumerate(colors(12)):
        ax_leg_angle.plot(
            times, joint_angle[:, i], color=color, label=f"joint {i}"
        )
    ax_leg_angle.legend(**legend_kwargs)

    #######
    # yaw #
    #######

    @jax.jit
    def yaw_fun(sol: utils.TableSol) -> jax.Array:
        yaw_cost_arr = opt.yaw_boundary_cost_arr(
            weights=weights,
            cost=cost_terms.yaw_cost,
            state=sol.x,
            control=sol.u,
        )
        yaw_cost_val = jnp.mean(yaw_cost_arr, axis=0)
        return yaw_cost_val

    yaw = jnp.array([yaw_fun(sol) for sol in trajectory])

    ax_yaw: mpl_ax.Axes = axes[2, 1]
    ax_yaw.set_title("Yaw boundary")
    ax_yaw.set_xlabel("[s]")
    ax_yaw.set_ylabel("[cost]")
    ax_yaw.set_xlim(times[0], times[-1])
    ax_yaw.set_ylim(*_get_limits(jnp.ravel(yaw)))
    ax_yaw.grid()

    ax_yaw.plot(times, yaw, color=colors(1)[0], label="yaw")
    ax_yaw.legend(**legend_kwargs)

    ###########
    # control #
    ###########

    @jax.jit
    def control_fun(sol: utils.TableSol) -> jax.Array:
        control_cost_arr = opt.control_cost_arr(
            weights=weights,
            control=sol.u,
        )
        control_cost_val = 0.5 * jnp.mean(control_cost_arr, axis=0)
        return control_cost_val

    control = jnp.array([control_fun(sol) for sol in trajectory])

    ax_control: mpl_ax.Axes = axes[3, 0]
    ax_control.set_title("Control")
    ax_control.set_xlabel("[s]")
    ax_control.set_ylabel("[cost]")
    ax_control.set_xlim(times[0], times[-1])
    ax_control.set_ylim(*_get_limits(jnp.ravel(control)))
    ax_control.grid()

    for i, color in enumerate(colors(6)):
        ax_control.plot(times, control[:, i], color=color, label=f"control {i}")
    ax_control.legend(**legend_kwargs)

    ###################
    # % contributions #
    ###################

    head_acc_terms = full_head_acc
    omega_terms = full_omega
    leg_pos_terms = jnp.sum(leg_pos, axis=1)
    leg_vel_terms = jnp.sum(leg_vel, axis=1)
    joint_angle_terms = jnp.sum(joint_angle, axis=1)
    yaw_terms = yaw
    control_terms = jnp.sum(control, axis=1)
    all_terms = [
        head_acc_terms,
        omega_terms,
        leg_pos_terms,
        leg_vel_terms,
        joint_angle_terms,
        yaw_terms,
        control_terms,
    ]

    costs = jnp.sum(jnp.stack(all_terms), axis=0)
    head_acc_perc = head_acc_terms / costs * 100.0
    omega_perc = omega_terms / costs * 100.0
    leg_pos_perc = leg_pos_terms / costs * 100.0
    leg_vel_perc = leg_vel_terms / costs * 100.0
    joint_angle_perc = joint_angle_terms / costs * 100.0
    yaw_perc = yaw_terms / costs * 100.0
    control_perc = control_terms / costs * 100.0
    all_perc = [
        head_acc_perc,
        omega_perc,
        leg_pos_perc,
        leg_vel_perc,
        joint_angle_perc,
        yaw_perc,
        control_perc,
    ]

    ax_perc: mpl_ax.Axes = axes[3, 1]
    ax_perc.set_title("Cost term percentage contributions")
    ax_perc.set_xlabel("[s]")
    ax_perc.set_ylabel("[%]")
    ax_perc.set_xlim(times[0], times[-1])
    ax_perc.set_ylim(0.0, 100.0)
    ax_perc.grid()

    labels = [
        "head acc %",
        "head omega %",
        "leg pos %",
        "leg vel %",
        "joint angle %",
        "yaw %",
        "control %",
    ]
    for data, label, color in zip(all_perc, labels, colors(len(labels))):
        ax_perc.plot(times, data, color=color, label=label)
    ax_perc.legend(**legend_kwargs)

    ##########
    # return #
    ##########

    return fig
