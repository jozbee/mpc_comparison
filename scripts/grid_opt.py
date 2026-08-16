"""Optimize mpc hyperparameters via grid search."""

import functools
import itertools
import multiprocessing as mp
import os
import pickle
import random

import control as ct
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp_mpc.stewart_min import mpc_spec, opt, siso, viz

jax.config.update("jax_enable_x64", True)


###########
# helpers #
###########


def load_clean_references(
    file_path: str, dt: float = mpc_spec.dt
) -> tuple[jax.Array, jax.Array]:
    """Load clean reference data, and do some light filtering."""
    data = np.array(pd.read_hdf(file_path))
    acc_ref = data[:, 1:4]
    omega_ref = data[:, 4:]

    # use 100 Hz sampling
    acc_ref = acc_ref[::2]
    omega_ref = omega_ref[::2]

    assert acc_ref.shape[0] == omega_ref.shape[0]
    assert acc_ref.shape[1] == 3
    assert omega_ref.shape[1] == 3

    # filt (butter)
    s = ct.tf("s") / (2 * np.pi * 5.0)
    butter = siso.DiscreteSISO.cont2discrete(
        1 / (1 + 2 * s + 2 * s**2 + s**3), dt=dt, method="bilinear"
    )
    ABCD = [butter.A, butter.B, butter.C, butter.D]

    def butter_int(u):
        assert len(u.shape) == 1
        one = np.ones(ABCD[0].shape[0])
        x0 = siso.obs_x0(*ABCD, y=one * u[0], u=one * u[0])
        return siso.lti_int(*ABCD, x0, u)[1]

    acc_ref = jnp.array([butter_int(u) for u in acc_ref.T]).T
    omega_ref = jnp.array([butter_int(u) for u in omega_ref.T]).T
    return acc_ref, omega_ref


@jax.jit
def get_omegas(ts: opt.TrainState) -> tuple[jax.Array, jax.Array]:
    return ts.y_vest_irl[0, 3:], ts.y_vest_sim[0, 3:]


@jax.jit
def get_accs(ts: opt.TrainState) -> tuple[jax.Array, jax.Array]:
    return ts.y_vest_irl[0, :3], ts.y_vest_sim[0, :3]


#######
# run #
#######


def single_sms(args: tuple) -> None:
    """Simple sms run.

    Saves 3 figures, and pickles parameters and some cost information.
    This function does not return any useful information.

    Parameters
    ----------
    args :
        Tuple with `(index, grid, path)`.
        (`grid` is specified in the cli specification at the bottom of the
        file, or you can look at 'setup' below.)
    """
    #########
    # setup #
    #########

    assert len(args) == 4
    index, grid, path, ref_file_path = args
    print(f"start: {index}\ngrid: {grid}\n")

    assert len(grid) == 6
    acc_weights = grid[0]
    omega_weights = grid[1]
    ctrl_weights = grid[2]
    alpha_acc = grid[3]
    alpha_omega = grid[4]
    n = grid[5]  # horizon_num

    acc_ref, omega_ref = load_clean_references(ref_file_path)
    assert acc_ref.shape[0] == omega_ref.shape[0]
    assert acc_ref.shape[1] == 3
    assert omega_ref.shape[1] == 3

    limits = mpc_spec.MPCLimits()
    spec = mpc_spec.MPCSpec()

    begin = 0
    num_steps = acc_ref.shape[0]

    weights = mpc_spec.ExpWeights(
        lin_dyn=jnp.array(acc_weights),
        omega=jnp.array(omega_weights),
        alpha_acc=jnp.array([alpha_acc]),
        alpha_omega=jnp.array([alpha_omega]),
        control=jnp.array(ctrl_weights),
    )
    limits = mpc_spec.MPCLimits()
    spec = mpc_spec.MPCSpec.init_weight_margins(
        weights, limits, max_iter=2, max_ls=1, use_terminal=True
    )
    train_step = functools.partial(
        opt.train_step_with_cost,
        spec,
    )

    #######
    # run #
    #######

    train_state = opt.TrainState.zero_init(spec, acc_ref[0, 3])
    train_list = []
    times = []
    res_list = []
    for i in range(num_steps):
        train_state, res, t_tot = train_step(
            train_state,
            acc_ref[begin + i],
            omega_ref[begin + i],
        )
        train_list.append(train_state)
        res_list.append(res)
        times.append(t_tot)

    #########
    # plots #
    #########

    trajectory = train_list
    references = {
        "xyz-acceleration": jnp.array(acc_ref[begin : begin + num_steps]),
        "angular-velocity": jnp.array(omega_ref[begin : begin + num_steps]),
    }

    mpc_human_fig = viz.plot_human_trajectory(
        trajectory=trajectory,
        limits=limits,
        spec=spec,
        references=references,
    )
    mpc_vestibular_fig = viz.plot_vestibular_trajectory(
        trajectory=trajectory,
        limits=limits,
        spec=spec,
    )
    mpc_actuator_fig = viz.plot_actuator_trajectory(
        trajectory=trajectory,
        limits=limits,
        spec=spec,
    )

    mpc_human_fig.savefig(f"{path}/{index}_human.png", dpi=300)
    mpc_vestibular_fig.savefig(f"{path}/{index}_vestibular.png", dpi=300)
    mpc_actuator_fig.savefig(f"{path}/{index}_actuator.png", dpi=300)

    plt.close(mpc_human_fig)
    plt.close(mpc_vestibular_fig)
    plt.close(mpc_actuator_fig)

    #########
    # error #
    #########

    omegas = [get_omegas(sol) for sol in trajectory]
    omega_irl = jnp.array([omega[0] for omega in omegas])
    omega_sim = jnp.array([omega[1] for omega in omegas])

    accs = [get_accs(sol) for sol in trajectory]
    acc_irl = jnp.array([acc[0] for acc in accs])
    acc_sim = jnp.array([acc[1] for acc in accs])

    omega_diff = omega_irl - omega_sim
    acc_diff = acc_irl - acc_sim

    omega_err = 0.5 * jnp.sum(omega_diff**2)
    acc_err = 0.5 * jnp.sum(acc_diff**2)
    tot_err = omega_err + acc_err

    omega_err_4 = 0.5 * jnp.sum(omega_diff**4)
    acc_err_4 = 0.5 * jnp.sum(acc_diff**4)
    tot_err_4 = omega_err_4 + acc_err_4

    ##########
    # pickle #
    ##########

    res = {
        "weights": weights,
        "cost_terms": spec.cost_terms,
        "horizon_length": n,
        "omega_err": omega_err,
        "acc_err": acc_err,
        "tot_err": tot_err,
        "omega_err_4": omega_err_4,
        "acc_err_4": acc_err_4,
        "tot_err_4": tot_err_4,
        "omega_diff": omega_diff,
        "acc_diff": acc_diff,
    }
    with open(f"{path}/{index}_params.pickle", "wb") as f:
        pickle.dump(res, f)

    print(f"done: {index}")


#######
# cli #
#######

if __name__ == "__main__":
    random.seed(42)

    ##########
    # params #
    ##########

    # file_name = "../data/clean_00_sms_drive.hdf"
    file_name = "../data/clean_specific-forces-lander_motion_redes_manual.hdf"
    save_dir = "./grid_data"

    acc_grid = [
        # np.ones(3) * 1e2,  # x, y, z acc weights
        np.ones(3) * 1e3,
        np.ones(3) * 1e4,
        np.ones(3) * 1e5,
    ]
    omega_ones = jnp.array([1e0, 1e0, 1e2])  # weight z-vel more
    omega_grid = [
        # omega_ones * 1e3,  # x, y, z, ang vel weights
        # omega_ones * 5e3,
        omega_ones * 1e4,
        omega_ones * 5e4,
        omega_ones * 1e5,
        omega_ones * 5e5,
    ]
    ctrl_grid = [
        # np.ones(6) * 1e0,
        np.ones(6) * 1e-1,
        np.ones(6) * 1e-2,
        np.ones(6) * 1e-3,
    ]
    alpha_acc_grid = [0.0, 1.0, 2.0, 4.0]  # exponential decay factor, acc
    alpha_omega_grid = [0.0, 1.0, 2.0, 4.0]  # exp decay factor, ang vel
    horizon_grid = [200]

    #######
    # run #
    #######

    os.makedirs(save_dir, exist_ok=True)

    grid_terms = [acc_grid, omega_grid, ctrl_grid]
    grid_terms.extend([alpha_acc_grid, alpha_omega_grid, horizon_grid])
    grid = list(itertools.product(*grid_terms))
    random.shuffle(grid)  # in-place shuffle

    args = [(i, grid[i], save_dir, file_name) for i in range(len(grid))]

    cpu_count = mp.cpu_count() // 2 + 2  # == 10
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=cpu_count) as p:
        p.map(single_sms, args)
