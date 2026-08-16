"""Export MPC solver, for cpp.

From `mpc_comparison/cpp`, call `python3 src/mpc_export.py`
"""

import functools

import jax
import jax.numpy as jnp
from jax2exec.jax2exec import jax2exec

from exp_mpc.stewart_min import mpc_spec, opt

jax.config.update("jax_enable_x64", True)


def round(x):
    return jnp.int64(jnp.round(jnp.squeeze(x)))


class PersonelMode:
    UNSUITED: int = 0
    SPECIAL: int = 1
    UNSUITED_LEFT: int = 2
    UNSUITED_RIGHT: int = 3
    NONE: int = 4
    SUITED: int = 5
    data: jax.Array = jnp.array(
        [
            [0.248, -0.0626, -1.02e-05, 0.329],  # 0
            [0.227, -0.044, 0.0343, 0.289],  # 1
            [0.219, -0.0382, 0.0504, 0.27],  # 2
            [0.219, -0.0382, -0.0504, 0.27],  # 3
            [0.19, -0.0064, 0, 0.193],  # 4
            [0.296, -0.0374, 0, 0.411],  # 5
        ]
    )

    def process_idx(self, ts, spec, idx):
        idx = round(idx)
        data = self.data[idx]
        spec.m_rotary = data[0]
        spec.r_0_rotary = data[1:]

    @classmethod
    def init_spec(cls, spec):
        return cls()


class PredictionMode:
    CONSTANT: int = 0
    LANDER: int = 1

    lander_alpha: jax.Array = jnp.array([1.45, 1.6, 0.8, 0.85, 0.4, 0.4])
    lander_E: jax.Array = jnp.stack(
        [mpc_spec.triple_E(a, mpc_spec.dt) for a in lander_alpha]
    )
    lander_n: jax.Array = jnp.array([50, 46, 20, 42, 14, 30], dtype=jnp.int64)

    def process_idx(self, ts, spec, idx):
        idx = round(idx)

        def constant_pred():
            # setting `iter` to zero is a hack
            # note that the prediction horizon code doesn't start running in
            #  `opt.train_step_with_cost_jax` until after a few iterations have
            #  run (so that enough smooth history has been adopted)
            return (
                jnp.zeros_like(self.lander_E),
                jnp.zeros_like(self.lander_n),
                jnp.array(0, dtype=jnp.int64),
            )

        def lander_pred():
            return self.lander_E, self.lander_n, ts.iter

        pred_E, pred_n, iter = jax.lax.switch(
            idx,
            [constant_pred, lander_pred],
        )
        spec.pred_E = pred_E
        spec.pred_n = pred_n
        ts.iter = iter

    @classmethod
    def init_spec(cls, spec):
        lander_E = jnp.stack(
            [mpc_spec.triple_E(a, mpc_spec.dt) for a in cls.lander_alpha]
        )
        res = cls()
        res.lander_E = lander_E
        return res


class WeightMode:
    CONSTANT: int = 0
    LANDER_00: int = 1
    LANDER_01: int = 2

    constant_weights: mpc_spec.ExpWeights = mpc_spec.ExpWeights(
        lin_dyn=jnp.ones(3) * 1e4,
        omega=jnp.ones(3) * 1e4,
        control=jnp.ones(6) * 1e-1,
        alpha_acc=jnp.array([0.0]),
        alpha_omega=jnp.array([0.0]),
    )
    lander_weights_00: mpc_spec.ExpWeights = mpc_spec.ExpWeights(  # lander_acc
        lin_dyn=jnp.ones(3) * 1e5,
        omega=jnp.array([1e0, 1e0, 1e2]) * 1e5,
        control=jnp.ones(6) * 1e-3,
        alpha_acc=jnp.array([1.0]),
        alpha_omega=jnp.array([0.0]),
    )
    lander_weights_01: mpc_spec.ExpWeights = mpc_spec.ExpWeights(  # lander_ome
        lin_dyn=jnp.ones(3) * 1e5,
        omega=jnp.array([1e0, 1e0, 1e2]) * 5e5,
        control=jnp.ones(6) * 1e-1,
        alpha_acc=jnp.array([0.0]),
        alpha_omega=jnp.array([0.0]),
    )

    def process_idx(self, ts, spec, idx):
        idx = round(idx)
        spec.weights = jax.lax.switch(
            idx,
            [
                lambda: self.constant_weights,
                lambda: self.lander_weights_00,
                lambda: self.lander_weights_01,
            ],
        )

    @classmethod
    def init_spec(cls, spec):
        return cls()


def mpc_solver(
    spec: mpc_spec.MPCSpec,
    zero_ts: opt.TrainState,
    modes: tuple[PersonelMode, PredictionMode, WeightMode],
    # input
    personnel_mode: jax.Array,  # 0
    prediction_mode: jax.Array,  # 1
    weight_mode: jax.Array,  # 2
    acc_ref: jax.Array,  # 3
    omega_ref: jax.Array,  # 4
    last_control: jax.Array,  # 5
    prefilt0: jax.Array,  # 6
    filt0: jax.Array,  # 7
    vstate0_irl: jax.Array,  # 8
    vstate0_sim: jax.Array,  # 9
    y_vest_sim_hist: jax.Array,  # 10
    xyz_hist: jax.Array,  # 11
    yaw_hist: jax.Array,  # 12
    quat_hist: jax.Array,  # 13
    terminal_param: jax.Array,  # 14
    iter: jax.Array,  # 15
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
    personnel_mode = jnp.atleast_1d(personnel_mode)
    prediction_mode = jnp.atleast_1d(prediction_mode)
    weight_mode = jnp.atleast_1d(weight_mode)
    terminal_param = jnp.atleast_1d(terminal_param)
    iter = jnp.atleast_1d(iter)
    assert personnel_mode.shape == (1,)
    assert prediction_mode.shape == (1,)
    assert weight_mode.shape == (1,)
    assert acc_ref.shape == (3,)
    assert omega_ref.shape == (3,)
    assert last_control.shape == zero_ts.control.shape
    assert prefilt0.shape == zero_ts.prefilt0.shape
    assert filt0.shape == zero_ts.filt0.shape
    assert vstate0_irl.shape == zero_ts.vstate0_irl.shape
    assert vstate0_sim.shape == zero_ts.vstate0_sim.shape
    assert y_vest_sim_hist.shape == (zero_ts.y_vest_sim_hist.size,)
    assert xyz_hist.shape == (zero_ts.xyz_hist.size,)
    assert yaw_hist.shape == (zero_ts.yaw_hist.size,)
    assert quat_hist.shape == (zero_ts.quat_hist.size,)
    assert terminal_param.shape == (1,)
    assert iter.shape == (1,)

    # better namings
    mpc_spec = spec
    mpc_ts = zero_ts

    # update
    mpc_ts.control = last_control
    mpc_ts.prefilt0 = prefilt0
    mpc_ts.filt0 = filt0
    mpc_ts.vstate0_irl = vstate0_irl
    mpc_ts.vstate0_sim = vstate0_sim
    mpc_ts.y_vest_sim_hist = y_vest_sim_hist.reshape(
        zero_ts.y_vest_sim_hist.shape
    )
    mpc_ts.xyz_hist = xyz_hist.reshape(zero_ts.xyz_hist.shape)
    mpc_ts.yaw_hist = yaw_hist.reshape(zero_ts.yaw_hist.shape)
    mpc_ts.quat_hist = quat_hist.reshape(zero_ts.quat_hist.shape)
    mpc_ts.terminal_param = terminal_param[0]
    mpc_ts.iter = round(iter[0])

    modes_idx_iter = zip(
        modes, [personnel_mode[0], prediction_mode[0], weight_mode[0]]
    )
    for mode, idx in modes_idx_iter:
        mode.process_idx(mpc_ts, mpc_spec, idx)

    # result
    res = opt.train_step_with_cost_jax(mpc_spec, mpc_ts, acc_ref, omega_ref)
    res_ts = res[0]
    u_xyz = res_ts.y_pre[0, :3]
    u_yaw = jnp.atleast_1d(res_ts.y_pre[0, 3])
    u_tilt = res_ts.y_pre[0, 4:]
    return (
        u_xyz,  # 0
        u_yaw,  # 1
        u_tilt,  # 2
        res_ts.control,  # 3
        res_ts.prefilt0,  # 4
        res_ts.filt0,  # 5
        res_ts.vstate0_irl,  # 6
        res_ts.vstate0_sim,  # 7
        res_ts.y_vest_sim_hist.flatten(),  # 8
        res_ts.xyz_hist.flatten(),  # 9
        res_ts.yaw_hist.flatten(),  # 10
        res_ts.quat_hist.flatten(),  # 11
        jnp.atleast_1d(res_ts.terminal_param),  # 12
        iter + 1,  # 13
    )


if __name__ == "__main__":
    # setup
    limits = mpc_spec.MPCLimits()
    spec = mpc_spec.MPCSpec.init_weight_margins(
        WeightMode.constant_weights,
        limits,
        max_iter=2,
        max_ls=1,
        use_terminal=True,
        init_norm=1e-1,
    )
    zero_ts = opt.TrainState.zero_init(spec)
    modes = (
        PersonelMode.init_spec(spec),
        PredictionMode.init_spec(spec),
        WeightMode.init_spec(spec),
    )
    fun = functools.partial(mpc_solver, spec, zero_ts, modes)

    # dummy input for tracing fun
    # (acc_ref, omega_ref, state0, vstate0_irl, vstate0_sim, control0,
    #  last_control)
    def make_dummy(val) -> jax.ShapeDtypeStruct:
        if hasattr(val, "size"):
            val = val.size
        return jax.ShapeDtypeStruct(shape=(val,), dtype=jnp.float64)

    dummy_in = (
        make_dummy(1),  # personnel mode, 0
        make_dummy(1),  # prediction mode, 1
        make_dummy(1),  # weight mode, 2
        make_dummy(3),  # acc_ref, 3
        make_dummy(3),  # omega_ref, 4
        make_dummy(zero_ts.control),  # 5
        make_dummy(zero_ts.prefilt0),  # 6
        make_dummy(zero_ts.filt0),  # 7
        make_dummy(zero_ts.vstate0_irl),  # 8
        make_dummy(zero_ts.vstate0_sim),  # 9
        make_dummy(zero_ts.y_vest_sim_hist),  # 10
        make_dummy(zero_ts.xyz_hist),  # 11
        make_dummy(zero_ts.yaw_hist),  # 12
        make_dummy(zero_ts.quat_hist),  # 13
        make_dummy(1),  # terminal param, 14
        make_dummy(1),  # iter, 15
    )

    # directory to save the compiled executable and metadata
    directory = "./artifacts"
    fun_name = "mpc_export"

    # compile the function and save the executable
    jax2exec(fun, dummy_in, directory, fun_name)
