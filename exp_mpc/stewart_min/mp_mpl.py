"""
Multi-processed matplotlib helpers.
We use the python ``multiprocessing`` library to call our animation routines in
parallel in several python processes.
Each python process only computes a portion of the animation, and then these
videos are stitched together with ``ffmpeg``.

**Warning**: This can use A LOT of RAM.
"""

import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
import numpy as np
import jax
import tempfile
import itertools
import pickle
import subprocess
import dataclasses

import exp_mpc.stewart_min.utils as utils
import exp_mpc.stewart_min.viz as viz
import exp_mpc.stewart_min.robo as robo


###########
# helpers #
###########


cpus: int = os.cpu_count()  # type: ignore
assert isinstance(cpus, int)
# cpus -= 4  # remove efficiency cpus, for laptop apple m-series processors
cpus //= 2  # jax threading, ram, and efficiency cpu considerations


def _get_frame_range_iter(
    trajectory: list[utils.TableSol],
    robo_params: robo.RoboParams,
    fps: int = 30,
    sim_rate: float = 1.0,
):
    num_frames = int(len(trajectory) * robo_params.dt * fps / sim_rate)
    frame_endpoints = [i * (num_frames // cpus) for i in range(cpus + 1)]
    frame_range_iter = zip(
        frame_endpoints[:-1],
        [point - 1 for point in frame_endpoints[1:]],
    )
    return frame_range_iter


def _concat_mp4(temp_dir: str, file_name: str, mp4_names: list[str]):
    temp_file_names = "".join([f"file '{name}'\n" for name in mp4_names])
    with open(f"{temp_dir}/temp_list.txt", "wt") as f:
        f.write(temp_file_names)

    # stitch back together
    # (note that we do not give the user progress updates, via DEVNULL)
    cmd = f"yes | ffmpeg -f concat -i {temp_dir}/temp_list.txt -c copy "
    cmd += f"{file_name}"
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    process.wait()  # need to wait; otherwise the temp dir kills itself


############################
# animate human trajectory #
############################


def _single_animate_human_trajectory(
    temp_dir: str,
    count: int,
    trajectory: list[utils.TableSol],
    references: dict[str, np.ndarray],
    frame_range: tuple[int, int],
    robo_params: robo.RoboParams,
) -> str:
    """Return file name for matplotlib animation."""
    anim, fig = viz.animate_human_trajectory(
        trajectory=trajectory,
        sim_rate=1.0,
        fps=30,
        references=references,
        frame_range=frame_range,
        robo_params=robo_params,
    )
    const_str = str(count)
    if len(const_str) == 1:
        const_str = f"0{const_str}"  # prepend an extra zero, for later sorting
    anim.save(f"{temp_dir}/{const_str}_temp.mp4", writer="ffmpeg", dpi=250)
    plt.close(fig)
    return f"{const_str}_temp.mp4"


@dataclasses.dataclass
class AnimateHumanTrajectoryArgs:
    """Arguments for :func:`mp_animate_human_trajectory`.

    Parameters
    ----------
    file_name :
        Destination mp4 file name.
    trajectory :
        Sequence of MPC solutions.
    references :
        Optional reference signals for the head plots.
        See :func:`exp_mpc.stewart_min.viz.animate_human_trajectory`.
    robo_params :
        Robot configuration.
    """

    file_name: str
    trajectory: list[utils.TableSol]
    references: dict[str, jax.Array]
    robo_params: robo.RoboParams


def mp_animate_human_trajectory(args: AnimateHumanTrajectoryArgs):
    """Render human trajectory animation in parallel and merge video chunks.

    Parameters
    ----------
    args :
        Bundle of inputs for the animation job.
        Includes output file name, MPC trajectory, optional references,
        and robot parameters.

    Notes
    -----
    The trajectory is split into frame ranges across worker processes.
    Each worker writes one temporary mp4 file.
    Final output is produced by concatenating those files with ``ffmpeg``.

    See Also
    --------
    :func:`exp_mpc.stewart_min.viz.animate_human_trajectory` :
        The underlying animation function, which is called in parallel across
        several processes.
    :func:`exp_mpc.stewart_min.mp_mpl.call_mp_animate_human_trajectory` :
        To call :func:`mp_animate_human_trajectory` in a jupyter notebook.
    """
    # setup
    temp_dir = tempfile.TemporaryDirectory()
    pool_inputs = zip(
        itertools.repeat(temp_dir.name),
        range(cpus),  # count_iter
        itertools.repeat(args.trajectory),
        itertools.repeat(args.references),
        _get_frame_range_iter(args.trajectory, robo_params=args.robo_params),
        itertools.repeat(args.robo_params),
    )

    # main
    with mp.Pool(cpus) as pool:
        names = pool.starmap(_single_animate_human_trajectory, pool_inputs)
    names = sorted(names)  # multiprocessing can mix things up
    _concat_mp4(
        temp_dir=temp_dir.name, file_name=args.file_name, mp4_names=names
    )


def call_mp_animate_human_trajectory(
    file_name: str,
    trajectory: list[utils.TableSol],
    references: dict[str, jax.Array],
    robo_params: robo.RoboParams,
):
    """Run :func:`mp_animate_human_trajectory` in a subprocess.

    To call :func:`mp_animate_human_trajectory` in a jupyter notebook, we need
    the extra level of indirection provided by this helper function.

    Parameters
    ----------
    file_name :
        Destination mp4 file name.
    trajectory :
        Sequence of MPC solutions.
    references :
        Optional reference signals for the head plots.
        Supported keys are ``"xyz-acceleration"`` and
        ``"angular-velocity"``.
    robo_params :
        Robot configuration.

    Notes
    -----
    Inputs are serialized to a temporary pickle file.
    This module is then invoked as a script with the corresponding CLI flag.
    The function blocks until subprocess completion.

    See Also
    --------
    :func:`exp_mpc.stewart_min.viz.animate_human_trajectory` :
        The underlying animation function, which is called in parallel across
        several processes.
    :func:`exp_mpc.stewart_min.mp_mpl.mp_animate_human_trajectory` :
        The main multi-processing function, which is called in a subprocess by
        this helper function.
    """
    args = AnimateHumanTrajectoryArgs(
        file_name=file_name,
        trajectory=trajectory,
        references=references,
        robo_params=robo_params,
    )
    temp_dir = tempfile.TemporaryDirectory()
    temp_pickle = f"{temp_dir.name}/mp_animate_human_trajectory_args.pickle"
    with open(temp_pickle, "wb") as f:
        pickle.dump(args, f)
    cmd = "source ~/.bash_profile && "
    cmd += f"python3 {__file__} --animate-human-trajectory-args {temp_pickle}"
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    process.wait()  # need to wait; otherwise the temp dir kills itself


######################
# animate trajectory #
######################


def _single_animate_trajectory(
    temp_dir: str,
    count: int,
    trajectory: list[utils.TableSol],
    frame_range: tuple[int, int],
    robo_params: robo.RoboParams,
    robo_geom: robo.RoboGeom,
) -> str:
    """Return file name for matplotlib animation."""
    anim, fig = viz.animate_trajectory(
        trajectory=trajectory,
        sim_rate=1.0,
        fps=30,
        frame_range=frame_range,
        robo_params=robo_params,
        robo_geom=robo_geom,
    )
    const_str = str(count)
    if len(const_str) == 1:
        const_str = f"0{const_str}"  # prepend an extra zero, for later sorting
    anim.save(f"{temp_dir}/{const_str}_temp.mp4", writer="ffmpeg", dpi=250)
    plt.close(fig)
    return f"{const_str}_temp.mp4"


@dataclasses.dataclass
class AnimateTrajectoryArgs:
    """Arguments for :func:`mp_animate_trajectory`.

    Parameters
    ----------
    file_name :
        Destination mp4 file name.
    trajectory :
        Sequence of MPC solutions.
    robo_params :
        Robot configuration.
    robo_geom :
        Stewart platform geometry.
    """

    file_name: str
    trajectory: list[utils.TableSol]
    robo_params: robo.RoboParams
    robo_geom: robo.RoboGeom


def mp_animate_trajectory(args: AnimateTrajectoryArgs):
    """Render table trajectory animation in parallel and merge video chunks.

    Parameters
    ----------
    args :
        Bundle of inputs for the animation job.
        Includes output file name, MPC trajectory, robot parameters,
        and robot geometry.

    Notes
    -----
    The trajectory is split into frame ranges across worker processes.
    Each worker writes one temporary mp4 file.
    Final output is produced by concatenating those files with ``ffmpeg``.

    See Also
    --------
    :func:`exp_mpc.stewart_min.viz.animate_trajectory` :
        The underlying animation function, which is called in parallel across
        several processes.
    :func:`exp_mpc.stewart_min.mp_mpl.call_mp_animate_trajectory` :
        To call :func:`mp_animate_trajectory` in a jupyter notebook.
    """
    # setup
    temp_dir = tempfile.TemporaryDirectory()
    pool_inputs = zip(
        itertools.repeat(temp_dir.name),
        range(cpus),  # count_iter
        itertools.repeat(args.trajectory),
        _get_frame_range_iter(args.trajectory, robo_params=args.robo_params),
        itertools.repeat(args.robo_params),
        itertools.repeat(args.robo_geom),
    )

    # main
    with mp.Pool(cpus) as pool:
        names = pool.starmap(_single_animate_trajectory, pool_inputs)
    names = sorted(names)  # multiprocessing can mix things up
    _concat_mp4(
        temp_dir=temp_dir.name, file_name=args.file_name, mp4_names=names
    )


def call_mp_animate_trajectory(
    file_name: str,
    trajectory: list[utils.TableSol],
    robo_params: robo.RoboParams,
    robo_geom: robo.RoboGeom,
):
    """Run :func:`mp_animate_trajectory` in a subprocess.

    Parameters
    ----------
    file_name :
        Destination mp4 file name.
    trajectory :
        Sequence of MPC solutions.
    robo_params :
        Robot configuration.
    robo_geom :
        Stewart platform geometry.

    Notes
    -----
    Inputs are serialized to a temporary pickle file.
    This module is then invoked as a script with the corresponding CLI flag.
    The function blocks until subprocess completion.

    See Also
    --------
    :func:`exp_mpc.stewart_min.viz.animate_trajectory` :
        The underlying animation function, which is called in parallel across
        several processes.
    :func:`exp_mpc.stewart_min.mp_mpl.mp_animate_trajectory` :
        The main multi-processing function, which is called in a subprocess by
        this helper function.
    """
    args = AnimateTrajectoryArgs(
        file_name=file_name,
        trajectory=trajectory,
        robo_params=robo_params,
        robo_geom=robo_geom,
    )
    temp_dir = tempfile.TemporaryDirectory()
    temp_pickle = f"{temp_dir.name}/mp_animate_trajectory_args.pickle"
    with open(temp_pickle, "wb") as f:
        pickle.dump(args, f)
    cmd = "source ~/.bash_profile && "
    cmd += f"python3 {__file__} --animate-trajectory-args {temp_pickle}"
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.wait()  # need to wait; otherwise the temp dir kills itself


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="mp_mpl",
        description="Run several matplotlib functions in parallel.",
    )

    parser.add_argument(
        "--animate-human-trajectory-args",
        type=str,
        help="Compute `viz.animate_human_trajectory` with multiprocessing.",
    )
    parser.add_argument(
        "--animate-trajectory-args",
        type=str,
        help="Compute `viz.animate_trajectory` with multiprocessing.",
    )

    args = parser.parse_args()

    if args.animate_human_trajectory_args:
        with open(args.animate_human_trajectory_args, "rb") as f:
            fun_args = pickle.load(f)
        mp_animate_human_trajectory(fun_args)
    elif args.animate_trajectory_args:
        with open(args.animate_trajectory_args, "rb") as f:
            fun_args = pickle.load(f)
        mp_animate_trajectory(fun_args)
    else:
        raise RuntimeError("Need to specify script action.")
