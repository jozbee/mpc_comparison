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
import tempfile
import itertools
import pickle
import subprocess

import exp_mpc.stewart_min.utils as utils
import exp_mpc.stewart_min.viz as viz
import exp_mpc.stewart_min.const as const


###########
# helpers #
###########


cpus: int = os.cpu_count()  # type: ignore
assert isinstance(cpus, int)
cpus -= 4  # remove efficiency cpus, for laptop apple m-series processors


def get_frame_range_iter(fps: int = 30, sim_rate: float = 1.0):
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


def mp_animate_human_trajectory(
    file_name: str,
    trajectory: list[utils.TableSol],
    references: dict[str, np.ndarray],
):
    # setup
    temp_dir = tempfile.TemporaryDirectory()
    pool_inputs = zip(
        itertools.repeat(temp_dir.name),
        range(cpus),  # count_iter
        itertools.repeat(trajectory),
        itertools.repeat(references),
        get_frame_range_iter(),
    )

    # main
    with mp.Pool(cpus) as pool:
        names = pool.starmap(single_animate_human_trajectory, pool_inputs)
    names = sorted(names)  # multiprocessing can mix things up
    concat_mp4(temp_dir=temp_dir.name, file_name=file_name, mp4_names=names)


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


def mp_animate_trajectory(
    file_name: str,
    trajectory: list[utils.TableSol],
):
    # setup
    temp_dir = tempfile.TemporaryDirectory()
    pool_inputs = zip(
        itertools.repeat(temp_dir.name),
        range(cpus),  # count_iter
        itertools.repeat(trajectory),
        get_frame_range_iter(),
    )

    # main
    with mp.Pool(cpus) as pool:
        names = pool.starmap(single_animate_trajectory, pool_inputs)
    names = sorted(names)  # multiprocessing can mix things up
    concat_mp4(temp_dir=temp_dir.name, file_name=file_name, mp4_names=names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="mp_mpl",
        description="Run several matplotlib functions in parallel.",
    )

    parser.add_argument(
        "--file-name",
        type=str,
        help="File name for result.",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        help="File path for pickled trajecotry data.",
    )
    parser.add_argument(
        "--references",
        type=str,
        help="File path for pickled references data.",
    )
    parser.add_argument(
        "--animate-human-trajectory",
        action="store_true",
        help="Compute `viz.animate_human_trajectory` with multiprocessing.",
    )
    parser.add_argument(
        "--animate-trajectory",
        action="store_true",
        help="Compute `viz.animate_trajectory` with multiprocessing.",
    )

    args = parser.parse_args()

    if args.animate_human_trajectory:
        with open(args.trajectory, "rb") as f:
            trajectory = pickle.load(f)
        with open(args.references, "rb") as f:
            references = pickle.load(f)
        mp_animate_human_trajectory(
            file_name=args.file_name,
            trajectory=trajectory,
            references=references,
        )
    elif args.animate_trajectory:
        with open(args.trajectory, "rb") as f:
            trajectory = pickle.load(f)
        mp_animate_trajectory(
            file_name=args.file_name,
            trajectory=trajectory,
        )
    else:
        raise RuntimeError("Need to specify script action.")
