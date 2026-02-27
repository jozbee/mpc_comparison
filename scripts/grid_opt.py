"""Optimize mpc hyperparameters via grid search."""

import random
import itertools
import functools
import pickle
import multiprocessing as mp

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

import exp_mpc.stewart_min.const as const
import exp_mpc.stewart_min.vest as vest
import exp_mpc.stewart_min.utils as utils
import exp_mpc.stewart_min.quartic_cost as quartic_cost
import exp_mpc.stewart_min.opt as opt
import exp_mpc.stewart_min.viz as viz

jax.config.update("jax_enable_x64", True)


###########
# helpers #
###########


def load_specific_sms_references(file_path: str) -> tuple[jax.Array, jax.Array]:
    df = pd.read_csv(file_path)

    ks = df.keys()

    ts = np.array(df[ks[0]])
    diff = np.abs(np.diff(ts))
    avg_diff = np.mean(diff)
    std_diff = np.std(diff)
    if std_diff > 0.05:
        bad_indices = np.where(diff > avg_diff + std_diff)[0] + 1  # off by one
        start_index = bad_indices[-2] + 5 * 200
        end_index = bad_indices[-1] - 1
    else:
        start_index = 0
        end_index = ts.size - 1

    df = df[start_index : end_index + 1]

    acc_ref = jnp.transpose(jnp.array([df[k] for k in ks[1:4]]))
    omega_ref = jnp.transpose(jnp.array([df[k] for k in ks[4:]]))
    return acc_ref, omega_ref


@jax.jit
def get_omegas(sol: utils.TableSol) -> tuple[jax.Array, jax.Array]:
    irl0 = sol.vstate_irl.get0()
    vstate0_irl = jnp.array([irl0.y_omegax, irl0.y_omegay, irl0.y_omegaz])
    sim0 = sol.vstate_sim.get0()
    vstate0_sim = jnp.array([sim0.y_omegax, sim0.y_omegay, sim0.y_omegaz])
    return vstate0_irl, vstate0_sim


@jax.jit
def get_accs(sol: utils.TableSol) -> tuple[jax.Array, jax.Array]:
    irl0 = sol.vstate_irl.get0()
    vstate0_irl = jnp.array([irl0.y_accx, irl0.y_accy, irl0.y_accz])
    sim0 = sol.vstate_sim.get0()
    vstate0_sim = jnp.array([sim0.y_accx, sim0.y_accy, sim0.y_accz])
    return vstate0_irl, vstate0_sim


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

    assert len(args) == 3
    index, grid, path = args
    print(f"start: {index}\ngrid: {grid}\n")

    assert len(grid) == 5
    acc_weights = grid[0]
    omega_weights = grid[1]
    alpha_acc = grid[2]
    alpha_omega = grid[3]
    n = grid[4]  # horizon_num

    ref_file_path = "/Users/jozbee/work/eng/comp/data/sms_00_sms_drive.csv"
    acc_ref, omega_ref = load_specific_sms_references(ref_file_path)
    assert acc_ref.shape[0] == omega_ref.shape[0]
    assert acc_ref.shape[1] == 3
    assert omega_ref.shape[1] == 3

    begin = 0
    num_steps = acc_ref.shape[0]

    weights = opt.ExpWeights(
        acc=jnp.array(acc_weights),
        omega=jnp.array(omega_weights),
        alpha_acc=jnp.array([alpha_acc]),
        alpha_omega=jnp.array([alpha_omega]),
    )

    margins = [0.2, 0.1]
    sizes = [1.0, 2**3, 2**8]
    euler_margins = [0.2 / 3.0, 0.1 / 3.0]
    euler_sizes = [2**0, 2**3, 2**8]

    leg_cost = quartic_cost.QuarticCost.from_bounds(
        margins=[0.3, 0.2, 0.1],
        sizes=[1.0, 2**3, 2**5, 2**10],
        low=const.leg_min,
        high=const.leg_max,
    )
    leg_vel_cost = quartic_cost.QuarticCost.from_bounds(
        margins=margins,
        sizes=sizes,
        low=-const.max_leg_vel,
        high=const.max_leg_vel,
    )
    joint_angle_cost = quartic_cost.QuarticCost.from_bounds(
        margins=margins,
        sizes=sizes,
        low=-const.joint_max_angle,
        high=const.joint_max_angle,
    )
    roll_cost = quartic_cost.QuarticCost.from_bounds(
        margins=euler_margins,
        sizes=euler_sizes,
        low=-const.max_roll,
        high=const.max_roll,
    )
    pitch_cost = quartic_cost.QuarticCost.from_bounds(
        margins=euler_margins,
        sizes=euler_sizes,
        low=-const.max_pitch,
        high=const.max_pitch,
    )
    yaw_cost = quartic_cost.QuarticCost.from_bounds(
        margins=euler_margins,
        sizes=euler_sizes,
        low=-const.max_rotary_yaw,
        high=const.max_rotary_yaw,
    )
    yaw_dot_cost = quartic_cost.QuarticCost.from_bounds(
        margins=euler_margins,
        sizes=euler_sizes,
        low=-const.max_rotary_vel,
        high=const.max_rotary_vel,
    )
    cost_terms = opt.CostTerms(
        leg_cost=leg_cost,
        leg_vel_cost=leg_vel_cost,
        joint_angle_cost=joint_angle_cost,
        roll_cost=roll_cost,
        pitch_cost=pitch_cost,
        yaw_cost=yaw_cost,
        yaw_dot_cost=yaw_dot_cost,
    )

    dt = const.dt
    dt_mpc = const.dt * 2.0
    tf_acc = vest.spec_refs["acc0"][0]
    tf_omega = vest.spec_refs["omega0"][0]
    vspec_acc = vest.VSpec.transfer2vspec(tf_acc, dt=dt, earth_moon_v0=True)
    vspec_omega = vest.VSpec.transfer2vspec(tf_omega, dt=dt)
    vspec_acc_mpc = vest.VSpec.transfer2vspec(
        tf_acc, dt=dt_mpc, earth_moon_v0=True
    )
    vspec_omega_mpc = vest.VSpec.transfer2vspec(tf_omega, dt=dt_mpc)

    train_step = functools.partial(
        opt.train_step_with_cost,
        weights,
        cost_terms,
        dt=dt,
        dt_mpc=dt_mpc,
        opt_options={"maxiter": 3, "maxls": 1},
        vspec_acc=vspec_acc,
        vspec_omega=vspec_omega,
        vspec_acc_mpc=vspec_acc_mpc,
        vspec_omega_mpc=vspec_omega_mpc,
        unroll=False,
        use_terminal=True,
    )

    #######
    # run #
    #######

    train_state = opt.TrainState.zero_init(
        horizon_num=n,
        vspec_acc=vspec_acc,
        vspec_omega=vspec_omega,
        vstate0_mode=("earth", "moon"),
    )
    train_list = []
    times = []
    sol_list = []
    res_list = []

    # for i in tqdm.tqdm(range(num_steps)):
    for i in range(num_steps):
        train_state, sol, res, t_tot = train_step(
            jnp.tile(acc_ref[begin + i], (n, 1)),
            jnp.tile(omega_ref[begin + i], (n, 1)),
            train_state,
        )
        train_list.append(train_state)
        sol_list.append(sol)
        res_list.append(res)
        times.append(t_tot)

    #########
    # plots #
    #########

    trajectory = sol_list
    references = {
        "xyz-acceleration": jnp.array(acc_ref[begin : begin + num_steps]),
        "angular-velocity": jnp.array(omega_ref[begin : begin + num_steps]),
    }

    mpc_human_fig = viz.plot_human_trajectory(
        trajectory=trajectory, references=references
    )
    mpc_vestibular_fig = viz.plot_vestibular_trajectory(trajectory=trajectory)
    mpc_actuator_fig = viz.plot_actuator_trajectory(trajectory=trajectory)

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
    acc_diff = acc_diff[:, :2]  # ignore the z_component

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
        "cost_terms": cost_terms,
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

    exp_scale = [1e2, 2e2, 3e2, 4e2]
    alpha_scale = [0.0, 1.0, 2.0, 4.0]

    acc_grid = [[1e2, 1e2, 1e0]]
    omega_grid = [[s, s, s] for s in exp_scale]
    alpha_acc_grid = alpha_scale.copy()
    alpha_omega_grid = alpha_scale.copy()
    horizon_grid = [200]

    grid_terms = [acc_grid, omega_grid]
    grid_terms.extend([alpha_acc_grid, alpha_omega_grid, horizon_grid])
    grid = list(itertools.product(*grid_terms))
    random.shuffle(grid)  # in-place shuffle

    args = [(i, grid[i], "./grid_data") for i in range(len(grid))]

    start_index = 0
    cpu_count = mp.cpu_count() // 2 + 2  # == 10
    tot = len(args)
    part_size = tot // cpu_count
    assert start_index < part_size

    if start_index != 0:
        tmps = [
            args[part_size * i : part_size * (i + 1)] for i in range(cpu_count)
        ]
        tmps = [tmp[start_index:] for tmp in tmps]
        args = list(itertools.chain(*tmps))

    with mp.Pool(processes=cpu_count) as p:
        p.map(single_sms, args)
