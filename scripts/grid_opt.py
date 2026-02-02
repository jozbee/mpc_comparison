"""Optimize mpc hyperparameters via grid search."""

import time
import random
import itertools
import functools
import pickle
import multiprocessing as mp

import jax.numpy as jnp
import scipy.optimize as sci_opt
import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import exp_mpc.stewart_min.viz as viz
import exp_mpc.stewart_min.const as const
import exp_mpc.stewart_min.opt as opt
import exp_mpc.stewart_min.utils as utils
import exp_mpc.stewart_min.quartic_cost as quartic_cost

import lbfgs.lbfgs as lbfgs

import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_log_compiles", True)


###########
# helpers #
###########


def load_sms_references(file_path: str) -> tuple[jax.Array, jax.Array]:
    """Load reference linear accelerations and angular velocities."""
    df = pd.read_csv(file_path)

    acc_keys = [
        f"sesmt.md.merged_frame.xyz_acc[{i}] {{m/s^2}}" for i in range(3)
    ]
    omega_keys = [
        f"sesmt.md.merged_frame.ang_vel[{i}] {{rad/s}}" for i in range(3)
    ]
    gravity_keys = [
        f"sesmt.md.merged_frame.gravity[{i}] {{m/s^2}}" for i in range(3)
    ]

    acc_ref = jnp.array(df[acc_keys])
    omega_ref = jnp.array(df[omega_keys])
    gravity_ref = jnp.array(df[gravity_keys])

    # for some reason, data collection after a lot of nonsense data
    # we grab the data after we start recognizing nonzero (x) accelerations
    # note that using direct equality is desired here (and not jnp.isclose)
    offset = jnp.argmax(acc_ref[:, 0] != 0.0)

    # we need to cancel and then add back in the gravity vector
    acc_ref = acc_ref[offset:, :] - 2 * gravity_ref[offset:, :]
    omega_ref = omega_ref[offset:, :]

    return acc_ref, omega_ref


def cost(
    args: tuple[
        opt.TrainState, opt.Weights, opt.CostTerms, jax.Array, jax.Array
    ],
    control_flat: jax.Array,
) -> jax.Array:
    train_state, weights, cost_terms, acc_ref, omega_ref = args
    return opt.cost_flat_jax(
        weights=weights,
        cost_terms=cost_terms,
        acc_ref=acc_ref,
        omega_ref=omega_ref,
        rstate0=train_state.rstate0,
        vstate0_irl=train_state.vstate0_irl,
        vstate0_sim=train_state.vstate0_sim,
        control0=train_state.control0,
        control_flat=control_flat,
    )


cost_and_grad = jax.jit(jax.value_and_grad(cost, argnums=1))
lbfgs_jit = jax.jit(lbfgs.lbfgs)


def train_step_with_cost(
    n: int,
    weights: opt.Weights,
    cost_terms: opt.CostTerms,
    train_state: opt.TrainState,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    **kwargs,
) -> tuple[opt.TrainState, utils.TableSol, sci_opt.OptimizeResult, float]:
    acc_ref = jnp.ravel(acc_ref)
    omega_ref = jnp.ravel(omega_ref)
    assert acc_ref.size == 3
    assert omega_ref.size == 3

    acc_ref = jnp.tile(A=acc_ref, reps=(n, 1))
    omega_ref = jnp.tile(A=omega_ref, reps=(n, 1))

    ts = train_state
    opt_options = kwargs.get("opt_options", {"maxiter": 16, "maxls": 8})
    opt_params = lbfgs.OptParamsLBFGS(
        fun=cost_and_grad,
        max_iter=opt_options["maxiter"],
        max_ls=opt_options["maxls"],
        tol=1e-5,
        c1=1e-4,
        c2=0.9,
    )
    t0 = time.time()
    res = lbfgs_jit(
        opt_params=opt_params,
        x0=train_state.control_flat,
        fun_params=(ts, weights, cost_terms, acc_ref, omega_ref),
    )
    res[0].block_until_ready()
    t1 = time.time()
    t_tot = t1 - t0

    control = utils.Control.from_flat(res[0])
    rstate = utils.get_rstate(control, ts.rstate0)
    vstate_irl = utils.get_vstate_irl(
        rstate, control, ts.control0, ts.vstate0_irl
    )
    vstate_sim = utils.get_vstate(acc_ref, omega_ref, ts.vstate0_sim)
    table_sol = utils.TableSol(
        x=rstate,
        u=control,
        vstate_irl=vstate_irl,
        vstate_sim=vstate_sim,
        stats=utils.TableStats(
            time=jnp.squeeze(t_tot),
            status=jnp.array(0),
            cost=jnp.array(res[1]),
        ),
    )
    next_state = opt.TrainState(
        rstate0=rstate.state[1],
        vstate0_irl=vstate_irl.x_state[1],
        vstate0_sim=vstate_sim.x_state[1],
        control0=control.control[0],
        control_flat=control.flatten(),
    )
    return next_state, table_sol, res, t_tot


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

    assert len(grid) == 6
    acc_weights = grid[0]
    omega_xy_weights = grid[1]
    omega_z_weight = grid[2]
    alpha_acc = grid[3]
    alpha_omega = grid[4]
    n = grid[5]  # horizon_num

    ref_file_path = "/Users/jozbee/work/eng/comp/data/00_sms_drive.csv"
    acc_ref, omega_ref = load_sms_references(ref_file_path)
    assert acc_ref.shape[0] == omega_ref.shape[0]
    assert acc_ref.shape[1] == 3
    assert omega_ref.shape[1] == 3

    begin = 3000
    num_steps = 5000

    weights = opt.ExpWeights(
        acc=jnp.array(acc_weights),
        omega=jnp.array(omega_xy_weights + [omega_z_weight]),
        alpha_acc=jnp.array([alpha_acc]),
        alpha_omega=jnp.array([alpha_omega]),
    )

    margins = [0.2, 0.1]
    sizes = [1.0, 2**3, 2**8]

    leg_cost = quartic_cost.QuarticCost.from_bounds(
        margins=[0.3, 0.2, 0.1],
        sizes=[1.0, 2**3, 2**5, 2**10],
        # margins=margins,
        # sizes=sizes,
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
    yaw_cost = quartic_cost.QuarticCost.from_bounds(
        margins=margins,
        sizes=sizes,
        low=-const.max_yaw,
        high=const.max_yaw,
    )
    cost_terms = opt.CostTerms(
        leg_cost=leg_cost,
        leg_vel_cost=leg_vel_cost,
        joint_angle_cost=joint_angle_cost,
        yaw_cost=yaw_cost,
    )

    train_step = functools.partial(train_step_with_cost, n, weights, cost_terms)

    #######
    # run #
    #######

    train_state = opt.TrainState.zero_init(n)
    train_list = []
    times = []
    sol_list = []
    res_list = []

    # for i in tqdm.tqdm(range(num_steps)):
    for i in range(num_steps):
        train_state, sol, res, t_tot = train_step(
            train_state,
            acc_ref[begin + i],
            omega_ref[begin + i],
            opt_options={"maxiter": 2, "maxls": 4},
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

    mpc_human_fig.savefig(f"{path}/{index}_human.png", dpi=250)
    mpc_vestibular_fig.savefig(f"{path}/{index}_vestibular.png", dpi=250)
    mpc_actuator_fig.savefig(f"{path}/{index}_actuator.png", dpi=250)

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

    omega_err = 0.5 * jnp.sum(jax.vmap(jnp.dot)(omega_diff, omega_diff))
    acc_err = 0.5 * jnp.sum(jax.vmap(jnp.dot)(acc_diff, acc_diff))
    tot_err = omega_err + acc_err

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
    }
    with open(f"{path}/{index}_params.pickle", "wb") as f:
        pickle.dump(res, f)

    print(f"done: {index}")


#######
# cli #
#######

if __name__ == "__main__":
    random.seed(42)

    exp_scale = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3]
    alpha_scale = [-1.0, 0.0, 1.0, 2.0, 4.0]

    acc_grid = [[s, s, 1e0] for s in exp_scale]
    omega_xy_grid = [[s, s] for s in exp_scale]
    omega_z_grid = exp_scale.copy()
    alpha_acc_grid = alpha_scale.copy()
    alpha_omega_grid = alpha_scale.copy()
    horizon_grid = [200, 400, 800]

    grid_terms = [acc_grid, omega_xy_grid, omega_z_grid]
    grid_terms.extend([alpha_acc_grid, alpha_omega_grid, horizon_grid])
    grid = list(itertools.product(*grid_terms))
    random.shuffle(grid)  # in-place shuffle

    args = [(i, grid[i], "./grid_data") for i in range(len(grid))]

    start_index = 26
    cpu_count = mp.cpu_count() // 2
    tot = len(args)
    part_size = tot // cpu_count
    assert start_index < part_size

    tmps = [args[part_size * i: part_size * (i + 1)] for i in range(cpu_count)]
    tmps = [tmp[start_index:] for tmp in tmps]
    args = list(itertools.chain(*tmps))

    with mp.Pool(processes=cpu_count) as p:
        p.map(single_sms, args)
