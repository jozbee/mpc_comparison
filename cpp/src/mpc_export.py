"""Export MPC solver, for cpp.

From `mpc_comparison/cpp`, call `python3 src/mpc_export.py`
"""

import functools

import jax
import jax.numpy as jnp
from jax2exec.jax2exec import jax2exec

from exp_mpc.stewart_min import mpc_spec, opt

jax.config.update("jax_enable_x64", True)


def mpc_solver(
    spec: mpc_spec.MPCSpec,
    zero_ts: opt.TrainState,
    # input
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    prefilt0: jax.Array,
    filt0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    xyz_hist: jax.Array,
    yaw_hist: jax.Array,
    tilt_hist: jax.Array,
    last_control: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Wrapper that performs one step of the mpc problem.

    Returns
    -------
    The next (control_irl, new_last_control, vstate0_irl, vstate0_sim).
    The first six elements of `control_irl` are for the next robot call.
    The remaining return values should just be passed to the next mpc call.
    """
    assert acc_ref.shape == (3,)
    assert omega_ref.shape == (3,)
    assert prefilt0.shape == zero_ts.prefilt0.shape
    assert filt0.shape == zero_ts.filt0.shape
    assert vstate0_irl.shape == zero_ts.vstate0_irl.shape
    assert vstate0_sim.shape == zero_ts.vstate0_sim.shape
    assert xyz_hist.shape == (zero_ts.xyz_hist.size,)
    assert yaw_hist.shape == (zero_ts.yaw_hist.size,)
    assert tilt_hist.shape == (zero_ts.quat_hist.size,)
    assert last_control.shape == zero_ts.control.shape

    acc_ref = jnp.tile(A=acc_ref, reps=(spec.n, 1))
    omega_ref = jnp.tile(A=omega_ref, reps=(spec.n, 1))

    mpc_ts = zero_ts  # better naming, now
    mpc_ts.prefilt0 = prefilt0
    mpc_ts.filt0 = filt0
    mpc_ts.vstate0_irl = vstate0_irl
    mpc_ts.vstate0_sim = vstate0_sim
    mpc_ts.xyz_hist = xyz_hist.reshape(zero_ts.xyz_hist.shape)
    mpc_ts.yaw_hist = yaw_hist.reshape(zero_ts.yaw_hist.shape)
    mpc_ts.quat_hist = tilt_hist.reshape(zero_ts.quat_hist.shape)
    mpc_ts.control = last_control

    res_ts = opt.train_step_with_cost_jax(spec, mpc_ts, acc_ref, omega_ref)[0]
    u_xyz = res_ts.y_pre[0, :3]
    u_yaw = jnp.atleast_1d(res_ts.y_pre[0, 3])
    u_tilt = res_ts.y_pre[0, 4:]
    return (
        u_xyz,  # 0
        u_yaw,  # 1
        u_tilt,  # 2
        res_ts.prefilt0,  # 3
        res_ts.filt0,  # 4
        res_ts.vstate0_irl,  # 5
        res_ts.vstate0_sim,  # 6
        res_ts.xyz_hist.flatten(),  # 7
        res_ts.yaw_hist.flatten(),  # 8
        res_ts.quat_hist.flatten(),  # 9
        res_ts.control,  # 10
    )


if __name__ == "__main__":
    # setup
    weights = mpc_spec.ExpWeights(
        lin_dyn=jnp.ones(3) * 1e5,
        omega=jnp.ones(3) * 5e5,
        control=jnp.ones(6) * 1e-1,
        alpha_acc=jnp.array([1.0]),
        alpha_omega=jnp.array([1.0]),
    )
    limits = mpc_spec.MPCLimits()
    spec = mpc_spec.MPCSpec.init_weight_margins(
        weights, limits, max_iter=2, max_ls=1, use_terminal=True, init_norm=1e-1
    )
    zero_ts = opt.TrainState.zero_init(spec)
    fun = functools.partial(mpc_solver, spec, zero_ts)

    # dummy input for tracing fun
    # (acc_ref, omega_ref, state0, vstate0_irl, vstate0_sim, control0,
    #  last_control)
    def make_dummy(val) -> jax.ShapeDtypeStruct:
        if hasattr(val, "size"):
            val = val.size
        return jax.ShapeDtypeStruct(shape=(val,), dtype=jnp.float64)

    dummy_in = (
        make_dummy(3),
        make_dummy(3),
        make_dummy(zero_ts.prefilt0),
        make_dummy(zero_ts.filt0),
        make_dummy(zero_ts.vstate0_irl),
        make_dummy(zero_ts.vstate0_sim),
        make_dummy(zero_ts.xyz_hist),
        make_dummy(zero_ts.yaw_hist),
        make_dummy(zero_ts.quat_hist),
        make_dummy(zero_ts.control),
    )

    # directory to save the compiled executable and metadata
    directory = "./artifacts"
    fun_name = "mpc_export"

    # compile the function and save the executable
    jax2exec(fun, dummy_in, directory, fun_name)
