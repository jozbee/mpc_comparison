"""Export MPC solver, for cpp.

From `mpc_comparison/cpp`, call `python3 src/mpc_export.py`
"""

import jax

jax.config.update("jax_enable_x64", True)

import functools  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jax2exec.jax2exec import jax2exec  # noqa: E402
import lbfgs.lbfgs as lbfgs  # noqa: E402
import exp_mpc.stewart_min.opt as opt  # noqa: E402
import exp_mpc.stewart_min.utils as utils  # noqa: E402
import exp_mpc.stewart_min.comp as comp  # noqa: E402
import exp_mpc.stewart_min.quartic_cost as quartic_cost  # noqa: E402
import exp_mpc.stewart_min.const as const  # noqa: E402


def mpc_solver(
    max_iter: int,
    max_ls: int,
    tol: float,
    c1: float,
    c2: float,
    n: int,  # horizon
    weights: opt.Weights,
    cost_terms: opt.CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    rstate0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    control0: jax.Array,
    last_control: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Wrapper that performs one step of the mpc problem.

    Returns
    -------
    The next (control, vstate0_irl, vstate0_sim).
    The first six elements of `control` are for the next robot call.
    The remaining vstates should just be passed to the next mpc call.
    """
    assert acc_ref.shape == (3,)
    assert omega_ref.shape == (3,)
    assert rstate0.shape == (12,)
    assert last_control.shape == (n * 6,)

    #########
    # setup #
    #########

    acc_ref = jnp.tile(A=acc_ref, reps=(n, 1))
    omega_ref = jnp.tile(A=omega_ref, reps=(n, 1))

    def cost_and_grad(
        args: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        rstate0, vstate0_irl, vstate0_sim, control0 = args
        control_flat = x
        return opt.cost_and_grad_flat_jax(
            weights=weights,
            cost_terms=cost_terms,
            acc_ref=acc_ref,
            omega_ref=omega_ref,
            rstate0=rstate0,
            vstate0_irl=vstate0_irl,
            vstate0_sim=vstate0_sim,
            control0=control0,
            control_flat=control_flat,
            use_rotary=True,
        )

    opt_params = lbfgs.OptParamsLBFGS(
        fun=cost_and_grad,
        max_iter=max_iter,
        max_ls=max_ls,
        tol=tol,
        c1=c1,
        c2=c2,
    )

    #########
    # solve #
    #########

    control, _, _ = lbfgs.lbfgs(
        opt_params=opt_params,
        x0=last_control,
        fun_params=(rstate0, vstate0_irl, vstate0_sim, control0),
    )

    ##################
    # update vstates #
    ##################

    # need to update vstate0_irl and vstate0_sim before returning

    # get irl controls
    rstate = utils.RState(rstate0)
    acc_irl = utils.rot(rstate, use_xy=False).T @ (control0[:3] + const.gravity)
    omega_irl = utils.angle_vel(rstate)
    # acc_irl = jnp.ones(3)
    # omega_irl = jnp.ones(3)

    # partition
    a_num = 3 * const.E0_acc.shape[0]
    v0_irl_a = vstate0_irl[:a_num]
    v0_irl_w = vstate0_irl[a_num:]
    v0_sim_a = vstate0_sim[:a_num]
    v0_sim_w = vstate0_sim[a_num:]

    # compute
    v0_irl_a = comp.lti_int_single(
        const.E0_acc, const.E1_acc, v0_irl_a, acc_irl
    )
    v0_irl_w = comp.lti_int_single(
        const.E0_omega, const.E1_omega, v0_irl_w, omega_irl
    )
    v0_sim_a = comp.lti_int_single(
        const.E0_acc, const.E1_acc, v0_sim_a, acc_ref[0]
    )
    v0_sim_w = comp.lti_int_single(
        const.E0_omega, const.E1_omega, v0_sim_w, omega_ref[0]
    )

    # return
    v0_irl = jnp.concatenate([v0_irl_a, v0_irl_w])
    v0_sim = jnp.concatenate([v0_sim_a, v0_sim_w])
    return control, v0_irl, v0_sim

    # TODO: delete
    # tmp = jnp.concatenate([acc_ref[0], omega_ref[0], rstate0, control0])
    # last_control = last_control.at[:tmp.size].set(tmp)
    # return last_control, v0_irl, v0_sim


if __name__ == "__main__":
    # mpc params
    weights = opt.ExpWeights(
        acc=jnp.array([1e1, 1e1, 1e0]),
        omega=jnp.array([1e1, 1e1, 1e1]),
        alpha_acc=jnp.array([0.0]),
        alpha_omega=jnp.array([0.0]),
    )
    margins = [0.2, 0.1]
    sizes = [2**0, 2**3, 2**8]
    yaw_margins = [0.2 / 3.0, 0.1 / 3.0]
    yaw_sizes = [2**0, 2**3, 2**8]
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
    yaw_cost = quartic_cost.QuarticCost.from_bounds(
        margins=yaw_margins,
        sizes=yaw_sizes,
        low=-const.max_rotary_yaw,
        high=const.max_rotary_yaw,
    )
    yaw_dot_cost = quartic_cost.QuarticCost.from_bounds(
        margins=yaw_margins,
        sizes=yaw_sizes,
        low=-const.max_rotary_vel,
        high=const.max_rotary_vel,
    )
    cost_terms = opt.CostTerms(
        leg_cost=leg_cost,
        leg_vel_cost=leg_vel_cost,
        joint_angle_cost=joint_angle_cost,
        yaw_cost=yaw_cost,
        yaw_dot_cost=yaw_dot_cost,
    )
    n = 200

    # lbfgs params
    max_iter = 4
    max_ls = 1
    tol = 1e-5
    c1 = 1e-4
    c2 = 0.9

    # fun
    fun = functools.partial(
        mpc_solver,
        max_iter,
        max_ls,
        tol,
        c1,
        c2,
        n,
        weights,
        cost_terms,
        # acc_ref,
        # omega_ref,
        # state0,
        # vstate0_irl,
        # vstate0_sim,
        # control0,
        # last_control,
    )

    # dummy input for tracing fun
    # (acc_ref, omega_ref, state0, vstate0_irl, vstate0_sim, control0,
    #  last_control)
    dummy_in = (
        jax.ShapeDtypeStruct(shape=(3,), dtype=jax.numpy.float64),
        jax.ShapeDtypeStruct(shape=(3,), dtype=jax.numpy.float64),
        jax.ShapeDtypeStruct(shape=(2 * 6,), dtype=jax.numpy.float64),
        jax.ShapeDtypeStruct(shape=(15,), dtype=jax.numpy.float64),
        jax.ShapeDtypeStruct(shape=(15,), dtype=jax.numpy.float64),
        jax.ShapeDtypeStruct(shape=(6,), dtype=jax.numpy.float64),
        jax.ShapeDtypeStruct(shape=(n * 6,), dtype=jax.numpy.float64),
    )

    # directory to save the compiled executable and metadata
    directory = "./artifacts"
    fun_name = "mpc_export"

    # compile the function and save the executable
    jax2exec(fun, dummy_in, directory, fun_name)
