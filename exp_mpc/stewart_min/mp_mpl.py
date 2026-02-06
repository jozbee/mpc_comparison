"""Multi-processed matplotlib helpers.

Namely, we multiprocess matplotlib animation writers.
Basically, we run several of these on different threads, and then we stich the
videos together after-the fact with matplotlib.
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
import exp_mpc.stewart_min.const as const
import exp_mpc.stewart_min.opt as opt


###########
# helpers #
###########


cpus: int = os.cpu_count()  # type: ignore
assert isinstance(cpus, int)
# cpus -= 4  # remove efficiency cpus, for laptop apple m-series processors
cpus //= 2  # jax threading, ram, and efficiency cpu considerations


def get_frame_range_iter(
    trajectory: list[utils.TableSol], fps: int = 30, sim_rate: float = 1.0
):
    num_frames = int(len(trajectory) * const.dt * fps / sim_rate)
    frame_endpoints = [i * (num_frames // cpus) for i in range(cpus + 1)]
    frame_range_iter = zip(
        frame_endpoints[:-1],
        [point - 1 for point in frame_endpoints[1:]],
    )
    return frame_range_iter


def concat_mp4(temp_dir: str, file_name: str, mp4_names: list[str]):
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


def single_animate_human_trajectory(
    temp_dir: str,
    count: int,
    trajectory: list[utils.TableSol],
    references: dict[str, np.ndarray],
    frame_range: tuple[int, int],
) -> str:
    """Return file name for matplotlib animation."""
    anim, fig = viz.animate_human_trajectory(
        trajectory=trajectory,
        sim_rate=1.0,
        fps=30,
        references=references,
        frame_range=frame_range,
    )
    const_str = str(count)
    if len(const_str) == 1:
        const_str = f"0{const_str}"  # prepend an extra zero, for later sorting
    anim.save(f"{temp_dir}/{const_str}_temp.mp4", writer="ffmpeg", dpi=250)
    plt.close(fig)
    return f"{const_str}_temp.mp4"


@dataclasses.dataclass
class AnimateHumanTrajectoryArgs:
    file_name: str
    trajectory: list[utils.TableSol]
    references: dict[str, jax.Array]


def mp_animate_human_trajectory(args: AnimateHumanTrajectoryArgs):
    # setup
    temp_dir = tempfile.TemporaryDirectory()
    pool_inputs = zip(
        itertools.repeat(temp_dir.name),
        range(cpus),  # count_iter
        itertools.repeat(args.trajectory),
        itertools.repeat(args.references),
        get_frame_range_iter(args.trajectory),
    )

    # main
    with mp.Pool(cpus) as pool:
        names = pool.starmap(single_animate_human_trajectory, pool_inputs)
    names = sorted(names)  # multiprocessing can mix things up
    concat_mp4(
        temp_dir=temp_dir.name, file_name=args.file_name, mp4_names=names
    )


def call_mp_animate_human_trajectory(
    file_name: str,
    trajectory: list[utils.TableSol],
    references: dict[str, jax.Array],
):
    print("hello")
    args = AnimateHumanTrajectoryArgs(
        file_name=file_name,
        trajectory=trajectory,
        references=references,
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


def single_animate_trajectory(
    temp_dir: str,
    count: int,
    trajectory: list[utils.TableSol],
    frame_range: tuple[int, int],
) -> str:
    """Return file name for matplotlib animation."""
    anim, fig = viz.animate_trajectory(
        trajectory=trajectory,
        sim_rate=1.0,
        fps=30,
        frame_range=frame_range,
    )
    const_str = str(count)
    if len(const_str) == 1:
        const_str = f"0{const_str}"  # prepend an extra zero, for later sorting
    anim.save(f"{temp_dir}/{const_str}_temp.mp4", writer="ffmpeg", dpi=250)
    plt.close(fig)
    return f"{const_str}_temp.mp4"


@dataclasses.dataclass
class AnimateTrajectoryArgs:
    file_name: str
    trajectory: list[utils.TableSol]


def mp_animate_trajectory(args: AnimateTrajectoryArgs):
    # setup
    temp_dir = tempfile.TemporaryDirectory()
    pool_inputs = zip(
        itertools.repeat(temp_dir.name),
        range(cpus),  # count_iter
        itertools.repeat(args.trajectory),
        get_frame_range_iter(args.trajectory),
    )

    # main
    with mp.Pool(cpus) as pool:
        names = pool.starmap(single_animate_trajectory, pool_inputs)
    names = sorted(names)  # multiprocessing can mix things up
    concat_mp4(
        temp_dir=temp_dir.name, file_name=args.file_name, mp4_names=names
    )


def call_mp_animate_trajectory(
    file_name: str,
    trajectory: list[utils.TableSol],
):
    args = AnimateTrajectoryArgs(
        file_name=file_name,
        trajectory=trajectory,
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


###########################
# animate cost trajectory #
###########################


def single_animate_cost_trajectory(
    temp_dir: str,
    count: int,
    trajectory: list[utils.TableSol],
    acc_refs,
    omega_refs,
    weights,
    cost_terms,
    frame_range: tuple[int, int],
) -> str:
    """Return file name for matplotlib animation."""
    anim, fig = viz.animate_cost_trajectory(
        trajectory=trajectory,
        acc_refs=acc_refs,
        omega_refs=omega_refs,
        weights=weights,
        cost_terms=cost_terms,
        sim_rate=1.0,
        fps=30,
        frame_range=frame_range,
    )
    const_str = str(count)
    if len(const_str) == 1:
        const_str = f"0{const_str}"  # prepend an extra zero, for later sorting
    anim.save(f"{temp_dir}/{const_str}_temp.mp4", writer="ffmpeg", dpi=250)
    plt.close(fig)
    return f"{const_str}_temp.mp4"


@dataclasses.dataclass
class AnimateCostTrajectoryArgs:
    file_name: str
    trajectory: list[utils.TableSol]
    acc_refs: jax.Array
    omega_refs: jax.Array
    weights: opt.Weights
    cost_terms: opt.CostTerms


def mp_animate_cost_trajectory(args: AnimateCostTrajectoryArgs):
    # setup
    temp_dir = tempfile.TemporaryDirectory()
    pool_inputs = zip(
        itertools.repeat(temp_dir.name),
        range(cpus),  # count_iter
        itertools.repeat(args.trajectory),
        itertools.repeat(args.acc_refs),
        itertools.repeat(args.omega_refs),
        itertools.repeat(args.weights),
        itertools.repeat(args.cost_terms),
        get_frame_range_iter(args.trajectory),
    )

    # main
    with mp.Pool(cpus) as pool:
        names = pool.starmap(single_animate_cost_trajectory, pool_inputs)
    names = sorted(names)  # multiprocessing can mix things up
    concat_mp4(
        temp_dir=temp_dir.name, file_name=args.file_name, mp4_names=names
    )


def call_mp_animate_cost_trajectory(
    file_name: str,
    trajectory: list[utils.TableSol],
    acc_refs: jax.Array,
    omega_refs: jax.Array,
    weights: opt.Weights,
    cost_terms: opt.CostTerms,
):
    args = AnimateCostTrajectoryArgs(
        file_name=file_name,
        trajectory=trajectory,
        acc_refs=acc_refs,
        omega_refs=omega_refs,
        weights=weights,
        cost_terms=cost_terms,
    )
    temp_dir = tempfile.TemporaryDirectory()
    temp_pickle = f"{temp_dir.name}/mp_animate_cost_trajectory_args.pickle"
    with open(temp_pickle, "wb") as f:
        pickle.dump(args, f)
    cmd = "source ~/.bash_profile && "
    cmd += f"python3 {__file__} --animate-cost-trajectory-args {temp_pickle}"
    process = subprocess.Popen(
        cmd,
        shell=True,
        # cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
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
    parser.add_argument(
        "--animate-cost-trajectory-args",
        type=str,
        help="Compute `viz.animate_cost_trajectory` with multiprocessing.",
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
    elif args.animate_cost_trajectory_args:
        with open(args.animate_cost_trajectory_args, "rb") as f:
            fun_args = pickle.load(f)
        mp_animate_cost_trajectory(fun_args)
    else:
        raise RuntimeError("Need to specify script action.")
