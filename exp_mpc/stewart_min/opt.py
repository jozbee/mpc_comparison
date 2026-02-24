"""Scipy optimization for the Stewart platform MPC control.

Idea:
 * All computations are done in jax.
  * TODO(jozbee): use an optimizer that is compatible with jax.
 * We use scipy.optimize to do the optimization.
  * SciPy requires flat arrays, so we build jax-pytree compatible dataclasses
    for convenient bookkeeping.
 * We use the L-BFGS-B optimizer from scipy.optimize.
  * This was determined to be the best SciPy optimizer, after extensive testing
    with a fixed point tracking problem.
    (The other solvers would hang and require several seconds per mpc step.)
 * We use a single shooting method, because our double integrator model is so
   implistic.

Numpy array conventions:
We control acceleration.
For scipy, the accelerations are in the Cartesian coordinate order
    [x, y, z, roll, pitch, yaw],
for one giant flat array.
"""

from __future__ import annotations

import dataclasses
import functools
import time
import typing as tp

import numpy as np
import jax
import jax.numpy as jnp

import exp_mpc.stewart_min.const as const
import exp_mpc.stewart_min.utils as utils
import exp_mpc.stewart_min.quartic_cost as quartic_cost

import lbfgs.lbfgs as lbfgs

# lbfgs_res = (minimizer, value, gradient)
lbfgs_result: tp.TypeAlias = tuple[jax.Array, jax.Array, jax.Array]

# make sure to enable 64-bit precision for jax
# this is necessary for good performance
# use the following line when importing this library
jax.config.update("jax_enable_x64", True)


#####################
# weighting classes #
#####################


def _init_field(arr: jax.Array) -> jax.Array:
    """Initialize a field with the same shape as the input array."""
    return dataclasses.field(
        default_factory=lambda: arr,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Weights:
    """Cost function weights."""

    acc: jax.Array = _init_field(jnp.ones(3))
    omega: jax.Array = _init_field(jnp.ones(3))
    leg: jax.Array = _init_field(jnp.ones(6))
    leg_vel: jax.Array = _init_field(jnp.ones(6))
    joint_angle: jax.Array = _init_field(jnp.ones(12))
    roll: jax.Array = _init_field(jnp.ones(1))
    pitch: jax.Array = _init_field(jnp.ones(1))
    yaw: jax.Array = _init_field(jnp.ones(1))
    yaw_dot: jax.Array = _init_field(jnp.ones(1))
    control: jax.Array = _init_field(jnp.ones(6))

    def __post_init__(self) -> None:
        assert self.acc.ndim == 1
        assert self.acc.shape[0] == 3
        assert self.omega.ndim == 1
        assert self.omega.shape[0] == 3
        assert self.leg.ndim == 1
        assert self.leg.shape[0] == 6
        assert self.leg_vel.ndim == 1
        assert self.leg_vel.shape[0] == 6
        assert self.joint_angle.ndim == 1
        assert self.joint_angle.shape[0] == 12
        assert self.roll.ndim == 1
        assert self.roll.shape[0] == 1
        assert self.pitch.ndim == 1
        assert self.pitch.shape[0] == 1
        assert self.yaw.ndim == 1
        assert self.yaw.shape[0] == 1
        assert self.yaw_dot.ndim == 1
        assert self.yaw_dot.shape[0] == 1
        assert self.control.ndim == 1
        assert self.control.shape[0] == 6

        assert jnp.issubdtype(self.acc.dtype, jnp.floating)
        assert jnp.issubdtype(self.omega.dtype, jnp.floating)
        assert jnp.issubdtype(self.leg.dtype, jnp.floating)
        assert jnp.issubdtype(self.leg_vel.dtype, jnp.floating)
        assert jnp.issubdtype(self.joint_angle.dtype, jnp.floating)
        assert jnp.issubdtype(self.roll.dtype, jnp.floating)
        assert jnp.issubdtype(self.pitch.dtype, jnp.floating)
        assert jnp.issubdtype(self.yaw.dtype, jnp.floating)
        assert jnp.issubdtype(self.yaw_dot.dtype, jnp.floating)
        assert jnp.issubdtype(self.control.dtype, jnp.floating)

    def _time_scale(self, n: int, name: str) -> jax.Array:
        """Get time scale weights for flat array.

        See the `ExpWeights` class for a nontrivial implementation
        """
        # identity
        return jnp.ones(n, dtype=float)

    def scale_acc(self, n: int) -> jax.Array:
        """Get scale weights for flat acceleration array."""
        time_scale = self._time_scale(n, "acc")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.acc.size))
        val_scale = jnp.tile(self.acc, (n, 1))
        return time_scale * val_scale

    def scale_omega(self, n: int) -> jax.Array:
        """Get scale weights for flat angular velocity array."""
        time_scale = self._time_scale(n, "omega")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.omega.size))
        val_scale = jnp.tile(self.omega, (n, 1))
        return time_scale * val_scale

    def scale_leg(self, n: int) -> jax.Array:
        """Get scale weights for flat leg length array."""
        time_scale = self._time_scale(n, "leg")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.leg.size))
        val_scale = jnp.tile(self.leg, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_leg_vel(self, n: int) -> jax.Array:
        """Get scale weights for flat leg length array."""
        time_scale = self._time_scale(n, "leg_vel")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.leg_vel.size))
        val_scale = jnp.tile(self.leg_vel, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_joint_angle(self, n: int) -> jax.Array:
        """Get scale weights for flat joint angle array."""
        time_scale = self._time_scale(n, "joint_angle")
        time_scale = jnp.tile(
            time_scale.reshape(-1, 1), (1, self.joint_angle.size)
        )
        val_scale = jnp.tile(self.joint_angle, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_roll(self, n: int) -> jax.Array:
        """Get scale weights for flat roll angle array."""
        time_scale = self._time_scale(n, "roll")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.roll.size))
        val_scale = jnp.tile(self.roll, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_pitch(self, n: int) -> jax.Array:
        """Get scale weights for flat pitch angle array."""
        time_scale = self._time_scale(n, "pitch")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.pitch.size))
        val_scale = jnp.tile(self.pitch, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_yaw(self, n: int) -> jax.Array:
        """Get scale weights for flat yaw angle array."""
        time_scale = self._time_scale(n, "yaw")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.yaw.size))
        val_scale = jnp.tile(self.yaw, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_yaw_dot(self, n: int) -> jax.Array:
        """Get scale weights for flat yaw angle array."""
        time_scale = self._time_scale(n, "yaw_dot")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.yaw_dot.size))
        val_scale = jnp.tile(self.yaw_dot, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_control(self, n: int) -> jax.Array:
        """Get scale weights for flat control array."""
        time_scale = self._time_scale(n, "control")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.control.size))
        val_scale = jnp.tile(self.control, (n, 1))
        return jnp.ravel(time_scale * val_scale)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ExpWeights(Weights):
    alpha_acc: jax.Array = _init_field(jnp.ones(1) * 4.0)
    alpha_omega: jax.Array = _init_field(jnp.ones(1) * 4.0)
    alpha_leg: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_leg_vel: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_joint_angle: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_roll: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_pitch: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_yaw: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_yaw_dot: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_control: jax.Array = _init_field(jnp.ones(1) * 0.0)

    def _time_scale(self, n: int, name: str) -> jax.Array:
        """Get time scale weights for flat array."""
        # exponential decrease
        alpha_map = {
            "acc": self.alpha_acc,
            "omega": self.alpha_omega,
            "leg": self.alpha_leg,
            "leg_vel": self.alpha_leg_vel,
            "joint_angle": self.alpha_joint_angle,
            "roll": self.alpha_roll,
            "pitch": self.alpha_pitch,
            "yaw": self.alpha_yaw,
            "yaw_dot": self.alpha_yaw_dot,
            "control": self.alpha_control,
        }
        return jnp.exp(-jnp.arange(n, dtype=float) / n * alpha_map[name])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CostTerms:
    leg_cost: quartic_cost.QuarticCost
    leg_vel_cost: quartic_cost.QuarticCost
    joint_angle_cost: quartic_cost.QuarticCost
    roll_cost: quartic_cost.QuarticCost
    pitch_cost: quartic_cost.QuarticCost
    yaw_cost: quartic_cost.QuarticCost
    yaw_dot_cost: quartic_cost.QuarticCost


###################
# hyperbolic cost #
###################


def _hyper(x: jax.Array) -> jax.Array:
    """Hyperbolic cost function?

    WARNING: look at the implementation to see if this is just the identity
    function.
    Otherwise, when composed with the vector L^2 norm, this function is
    quadratic, but has the same linear asymptotics as the L^1 norm.

    Note
    ----
    The hyperbolic tangent function is modified to work better with auto-diff.
    The original function is

    .. math:: \\sqrt{1 + \frac{x^2}{a^2}} - 1.

    Instead, we use

    .. math:: \\sqrt{1 + \frac{x}{a}} - 1.

    So, instead of passing the L^2 norm, we should pass the squared L^2 norm.
    Even though the two forms are equivalent under these norm conventions,
    the compiler will _not_ cancel the square root and the square, which causes
    intermediate values to become nan.
    """
    # we can scale the input to make the quadratic region of attraction larger
    # for us, unity works quite well
    # a = 2**0
    # return jnp.sqrt(1.0 + x / a) - 1.0

    # identity...
    return x


########
# cost #
########


def acc_cost_arr(
    weights: Weights,
    vstate_irl: utils.VState,
    vstate_sim: utils.VState,
    control: utils.Control,
) -> jax.Array:
    """Head acceleration cost terms."""
    w = weights.scale_acc(control.size)
    a_irl = jnp.array([vstate_irl.y_accx, vstate_irl.y_accy, vstate_irl.y_accz])
    a_sim = jnp.array([vstate_sim.y_accx, vstate_sim.y_accy, vstate_sim.y_accz])
    diff = (a_irl - a_sim) * w.T
    diff_xy = diff[:2]
    diff_z = diff[2]

    def hyper(arr0, arr1):
        arr0 = jnp.atleast_1d(arr0)
        arr1 = jnp.atleast_1d(arr1)
        return _hyper(arr0 @ arr0) + _hyper(arr1 @ arr1)

    return jax.vmap(hyper)(diff_xy.T, diff_z)


def _acc_cost(
    weights: Weights,
    vstate_irl: utils.VState,
    vstate_sim: utils.VState,
    control: utils.Control,
) -> jax.Array:
    """Head acceleration cost."""
    cost_arr = acc_cost_arr(weights, vstate_irl, vstate_sim, control)
    return 0.5 * jnp.mean(cost_arr)


def omega_cost_arr(
    weights: Weights,
    vstate_irl: utils.VState,
    vstate_sim: utils.VState,
    control: utils.Control,
) -> jax.Array:
    """Angular velocity cost terms."""
    w = weights.scale_omega(control.size)
    w_irl = jnp.array(
        [vstate_irl.y_omegax, vstate_irl.y_omegay, vstate_irl.y_omegaz]
    )
    w_sim = jnp.array(
        [vstate_sim.y_omegax, vstate_sim.y_omegay, vstate_sim.y_omegaz]
    )
    diff = (w_irl - w_sim) * w.T
    diff_xy = diff[:2]
    diff_z = diff[2]

    def hyper(arr0, arr1):
        arr0 = jnp.atleast_1d(arr0)
        arr1 = jnp.atleast_1d(arr1)
        return _hyper(arr0 @ arr0) + _hyper(arr1 @ arr1)

    return jax.vmap(hyper)(diff_xy.T, diff_z)


def _omega_cost(
    weights: Weights,
    vstate_irl: utils.VState,
    vstate_sim: utils.VState,
    control: utils.Control,
) -> jax.Array:
    """Angular velocity cost."""
    cost_arr = omega_cost_arr(weights, vstate_irl, vstate_sim, control)
    return 0.5 * jnp.mean(cost_arr)


def leg_boundary_cost_arr(
    weights: Weights,
    length_cost: quartic_cost.QuarticCost,
    vel_cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
    use_rotary: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Include leg length and leg velocity costs.

    By using automatic differentiation, we can compute the lengths and the
    velocities cheaper than computing them separately, which is productive.
    """
    leg_pos_vel = functools.partial(utils.leg_pos_vel, use_rotary=use_rotary)
    lengths, vels = jax.vmap(leg_pos_vel)(state.pop0())
    lengths = jnp.ravel(lengths)
    vels = jnp.ravel(vels)
    length_costs = jax.vmap(length_cost)(lengths)
    vel_costs = jax.vmap(vel_cost)(vels)
    w_len = weights.scale_leg(control.size)
    w_vel = weights.scale_leg_vel(control.size)
    length_cost_arr = jnp.reshape(length_costs * w_len, shape=(-1, 6))
    vel_cost_arr = jnp.reshape(vel_costs * w_vel, shape=(-1, 6))
    return length_cost_arr, vel_cost_arr


def _leg_boundary_cost(
    weights: Weights,
    length_cost: quartic_cost.QuarticCost,
    vel_cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
    use_rotary: bool,
) -> jax.Array:
    """Include leg length and leg velocity costs.

    By using automatic differentiation, we can compute the lengths and the
    velocities cheaper than computing them separately, which is productive.
    """
    length_cost_arr, vel_cost_arr = leg_boundary_cost_arr(
        weights, length_cost, vel_cost, state, control, use_rotary
    )
    length_cost_val = jnp.sum(jnp.mean(length_cost_arr, axis=0))
    vel_cost_val = jnp.sum(jnp.mean(vel_cost_arr, axis=0))
    return length_cost_val + vel_cost_val


def joint_angles(state: utils.RState, use_rotary: bool) -> jax.Array:
    return jnp.concatenate(utils.angle_joint(state, use_rotary))


def joint_angle_boundary_cost_arr(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
    use_rotary: bool = True,
) -> jax.Array:
    """Joint angle cost.

    This is about 3 times more expensive to compute than the other
    cost functions (including boundary cost functions).
    """
    joint_angles_part = functools.partial(joint_angles, use_rotary=use_rotary)
    angles = jnp.ravel(jax.vmap(joint_angles_part)(state.pop0()))
    costs = jax.vmap(cost)(angles)
    w = weights.scale_joint_angle(control.size)
    return (costs * w).reshape(-1, 12)


def _joint_angle_boundary_cost(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
    use_rotary: bool,
) -> jax.Array:
    """Joint angle cost.

    This is about 3 times more expensive to compute than the other
    cost functions (including boundary cost functions).
    """
    cost_arr = joint_angle_boundary_cost_arr(
        weights, cost, state, control, use_rotary
    )
    return jnp.sum(jnp.mean(cost_arr, axis=0))


def roll_boundary_cost_arr(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    roll = state.roll[1:]
    costs = jax.vmap(cost)(roll)
    w = weights.scale_roll(control.size)
    return costs * w


def _roll_boundary_cost(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    cost_arr = roll_boundary_cost_arr(weights, cost, state, control)
    return jnp.mean(cost_arr)


def pitch_boundary_cost_arr(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    pitch = state.pitch[1:]
    costs = jax.vmap(cost)(pitch)
    w = weights.scale_pitch(control.size)
    return costs * w


def _pitch_boundary_cost(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    cost_arr = pitch_boundary_cost_arr(weights, cost, state, control)
    return jnp.mean(cost_arr)


def yaw_boundary_cost_arr(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    yaw = state.yaw[1:]
    costs = jax.vmap(cost)(yaw)
    w = weights.scale_yaw(control.size)
    return costs * w


def _yaw_boundary_cost(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    cost_arr = yaw_boundary_cost_arr(weights, cost, state, control)
    return jnp.mean(cost_arr)


def yaw_dot_boundary_cost_arr(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    yaw_dot = state.yaw_dot[1:]
    yaw_dot = jnp.ravel(jnp.transpose(yaw_dot))
    costs = jax.vmap(cost)(yaw_dot)
    w = weights.scale_yaw_dot(control.size)
    return costs * w


def _yaw_dot_boundary_cost(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    cost_arr = yaw_dot_boundary_cost_arr(weights, cost, state, control)
    return jnp.mean(cost_arr)


def control_cost_arr(
    weights: Weights,
    control: utils.Control,
) -> jax.Array:
    w = weights.scale_control(control.size)
    costs = jnp.square(control.flatten() * w)
    return costs.reshape(-1, 6)


def _control_cost(
    weights: Weights,
    control: utils.Control,
) -> jax.Array:
    cost_arr = control_cost_arr(weights, control)
    return 0.5 * jnp.sum(jnp.mean(cost_arr, axis=0))


def _terminal_cost(
    rstate: utils.RState,
    vstate_irl: utils.VState,
    vstate_sim: utils.VState,
) -> jax.Array:
    # setup
    vi = vstate_irl
    vs = vstate_sim

    a_P = const.vspec_acc_sim.P
    o_P = const.vspec_omega_sim.P

    x_accx0 = a_P @ vs.x_accx[0]
    x_accx1 = a_P @ vs.x_accx[-1]
    x_accy0 = a_P @ vs.x_accy[0]
    x_accy1 = a_P @ vs.x_accy[-1]
    x_accz0 = a_P @ vs.x_accz[0]
    x_accz1 = a_P @ vs.x_accz[-1]
    x_omegax0 = o_P @ vs.x_omegax[0]
    x_omegax1 = o_P @ vs.x_omegax[-1]
    x_omegay0 = o_P @ vs.x_omegay[0]
    x_omegay1 = o_P @ vs.x_omegay[-1]
    x_omegaz0 = o_P @ vs.x_omegaz[0]
    x_omegaz1 = o_P @ vs.x_omegaz[-1]

    irl_x_omegaz1 = o_P @ vi.x_omegaz[-1]
    o_diff = irl_x_omegaz1 - x_omegaz1

    def o_V(x):
        return jnp.dot(const.ome_V @ x, x)

    # compute

    def scale(x):
        return jnp.exp(-50.0 * jnp.sum(jnp.square(x)))

    vt_cost = o_V(o_diff) * (scale(x_omegaz0) * scale(x_omegaz1)) * 4.0

    scale0 = jnp.array(
        [
            scale(x_accx0),
            scale(x_accy0),
            scale(x_accz0),
            scale(x_omegax0),
            scale(x_omegay0),
            scale(x_omegaz0),
            scale(x_accx0),
            scale(x_accy0),
            scale(x_accz0),
            scale(x_omegax0),
            scale(x_omegay0),
            scale(x_omegaz0),
        ]
    )
    scale1 = jnp.array(
        [
            scale(x_accx1),
            scale(x_accy1),
            scale(x_accz1),
            scale(x_omegax1),
            scale(x_omegay1),
            scale(x_omegaz1),
            scale(x_accx1),
            scale(x_accy1),
            scale(x_accz1),
            scale(x_omegax1),
            scale(x_omegay1),
            scale(x_omegaz1),
        ]
    )

    scales = scale0 * scale1 * 0.2
    rt_cost = jnp.sum(jnp.square(rstate.state[-1]) * scales)
    rt_cost += jnp.square(rstate.state[-1][5]) * (
        scale(x_omegaz0) * scale(x_omegaz1) * 2.0
    )

    return rt_cost + vt_cost


def _cost(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    rstate0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    control0: jax.Array,
    control: utils.Control,
    use_rotary: bool = True,  # static
    use_terminal: bool = True,
) -> jax.Array:
    # precompute states
    rstate, vstate_irl, vstate_sim = utils.get_states_with_eigen(
        const.dt_sim,
        const.vspec_acc_sim,
        const.vspec_omega_sim,
        acc_ref,
        omega_ref,
        rstate0,
        vstate0_irl,
        vstate0_sim,
        control0,
        control,
    )

    # cost
    cost = jnp.array(0.0)
    cost += _acc_cost(weights, vstate_irl, vstate_sim, control)
    cost += _omega_cost(weights, vstate_irl, vstate_sim, control)
    cost += _leg_boundary_cost(
        weights,
        cost_terms.leg_cost,
        cost_terms.leg_vel_cost,
        rstate,
        control,
        use_rotary,
    )
    cost += _joint_angle_boundary_cost(
        weights, cost_terms.joint_angle_cost, rstate, control, use_rotary
    )
    cost += _roll_boundary_cost(weights, cost_terms.roll_cost, rstate, control)
    cost += _pitch_boundary_cost(
        weights, cost_terms.pitch_cost, rstate, control
    )
    cost += _yaw_boundary_cost(weights, cost_terms.yaw_cost, rstate, control)
    cost += _yaw_dot_boundary_cost(
        weights, cost_terms.yaw_dot_cost, rstate, control
    )
    cost += _control_cost(weights, control)
    if use_terminal:
        cost += _terminal_cost(rstate, vstate_irl, vstate_sim)
    return cost


#######################
# scipy cost wrappers #
#######################


@functools.partial(jax.jit, static_argnames=["use_rotary", "use_terminal"])
def cost_flat_jax(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    rstate0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    control0: jax.Array,
    control_flat: jax.Array,
    use_rotary: bool = True,
    use_terminal: bool = True,
) -> jax.Array:
    control = utils.Control.from_flat(control_flat)
    return _cost(
        weights,
        cost_terms,
        acc_ref,
        omega_ref,
        rstate0,
        vstate0_irl,
        vstate0_sim,
        control0,
        control,
        use_rotary,
        use_terminal,
    )


@functools.partial(jax.jit, static_argnames=["use_rotary", "use_terminal"])
def cost_and_grad_flat_jax(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    rstate0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    control0: jax.Array,
    control_flat: jax.Array,
    use_rotary: bool = True,
    use_terminal: bool = True,
) -> tuple[jax.Array, jax.Array]:
    cost = functools.partial(
        cost_flat_jax,
        weights,
        cost_terms,
        acc_ref,
        omega_ref,
        rstate0,
        vstate0_irl,
        vstate0_sim,
        control0,
        use_rotary=use_rotary,
        use_terminal=use_terminal,
    )
    cost_and_grad = jax.value_and_grad(cost)
    return cost_and_grad(control_flat)


def get_scipy_cost(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    rstate0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    control0: jax.Array,
    use_rotary: bool = True,
) -> tuple[tp.Callable, tp.Callable]:
    """Get scipy cost functions for L-BFGS-B.

    WARNING: we cheat with the gradient computations.
    We always assume that the cost is querried before the cost gradient is
    querried, cf. `cost_jac_wrapper` in the implementation.
    This is reasonable and efficient for teh L-BFGS-B solver in scipy, but it
    is a dangerous hack, in general.

    Parameters
    ----------
    acc_ref :
        Reference head acceleration.
        Should have shape (n, 3) with n the number of control points.
    cost_terms :
        Quartic costs to scale for (inequality) boundary conditions.
    omega_ref :
        Reference head angular velocity.
        Should have shape (n, 3) with n the number of control points.
    rstate0 :
        Initial state, as a vector of size 12.
    vstate0_irl :
        Previous real-life vestibular state.
        (Take the previous so that rstate0 and control0 and provide feedback.)
    vstate0_sim :
        Current simulation vestibular state
    use_rotary :
        True if there is a rotary top, and false otherwise.

    Returns
    -------
    Cost function, gradient function
    """
    params = (
        weights,
        cost_terms,
        acc_ref,
        omega_ref,
        rstate0,
        vstate0_irl,
        vstate0_sim,
        control0,
    )

    cost_and_grad = functools.partial(
        cost_and_grad_flat_jax, *params, use_rotary=use_rotary
    )

    # history (use a list to mimic pointers)
    cost_mem: list[np.ndarray] = [np.array(np.nan)]
    grad_mem: list[np.ndarray] = [np.array(np.nan)]

    def update_mem(control_flat: np.ndarray) -> None:
        val, grad = cost_and_grad(control_flat)
        cost_mem[0] = np.array(val)
        grad_mem[0] = np.array(grad)

    def cost_wrapper(control_flat: np.ndarray) -> float:
        # warning: not safe for general use
        update_mem(control_flat)
        return float(cost_mem[0])

    def cost_jac_wrapper(control_flat: np.ndarray) -> np.ndarray:
        # warning: not safe for general use
        return grad_mem[0]

    return cost_wrapper, cost_jac_wrapper


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TrainState:
    """State (aux info) for training routine."""

    rstate0: jax.Array
    vstate0_irl: jax.Array
    vstate0_sim: jax.Array
    control0: jax.Array
    control_flat: jax.Array

    @classmethod
    def zero_init(
        cls, horizon_num: int, vstate0_mode: tp.Optional[tuple[str, str]] = None
    ) -> "TrainState":
        """Init train state with zeros.

        Parameters
        ----------
        horizon_num :
            Number of time steps in horizon length.
        vstate0_mode :
            Determines if the initial vestibular states should be initialized
            to respect gravity.
            The first and second entries represent the irl and sim conditions,
            respectively.
            If not `None`, the options are "earth" or "moon"

        Returns
        -------
        Zeroed train state.
        """
        acc_num = const.vspec_acc.E0.shape[0]
        omega_num = const.vspec_omega.E0.shape[0]
        v0_earth = const.vspec_acc.v0_earth
        v0_moon = const.vspec_acc.v0_moon
        u_num = 6
        r_num = u_num * 2
        v_num = 3 * acc_num + 3 * omega_num

        # gravity_range and gravity_map (setup)
        gr = (2 * acc_num, 3 * acc_num)
        g_map = {"earth": v0_earth, "moon": v0_moon}

        def zeros(n):
            return jnp.zeros(n, dtype=float)

        vstate0_irl = zeros(v_num)
        vstate0_sim = zeros(v_num)
        if vstate0_mode is not None:
            vstate0_irl = vstate0_irl.at[gr[0] : gr[1]].set(
                g_map[vstate0_mode[0]]
            )
            vstate0_sim = vstate0_sim.at[gr[0] : gr[1]].set(
                g_map[vstate0_mode[1]]
            )

        return cls(
            rstate0=zeros(r_num),
            vstate0_irl=vstate0_irl,
            vstate0_sim=vstate0_sim,
            control0=zeros(u_num),
            control_flat=zeros(u_num * horizon_num),
        )


def get_scipy_cost_ts(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    train_state: TrainState,
    use_rotary: bool = True,
) -> tuple[tp.Callable, tp.Callable]:
    """Wrapper for get_scipy_cost, but uses TrainState structure"""
    return get_scipy_cost(
        weights=weights,
        cost_terms=cost_terms,
        acc_ref=acc_ref,
        omega_ref=omega_ref,
        rstate0=train_state.rstate0,
        vstate0_irl=train_state.vstate0_irl,
        vstate0_sim=train_state.vstate0_sim,
        control0=train_state.control0,
        use_rotary=use_rotary,
    )


################
# jax training #
################


def lbfgs_cost(
    args: tuple[TrainState, Weights, CostTerms, jax.Array, jax.Array],
    control_flat: jax.Array,
) -> jax.Array:
    train_state, weights, cost_terms, acc_ref, omega_ref = args
    return cost_flat_jax(
        weights=weights,
        cost_terms=cost_terms,
        acc_ref=acc_ref,
        omega_ref=omega_ref,
        rstate0=train_state.rstate0,
        vstate0_irl=train_state.vstate0_irl,
        vstate0_sim=train_state.vstate0_sim,
        control0=train_state.control0,
        control_flat=control_flat,
        use_rotary=True,
    )


lbfgs_cost_and_grad = jax.jit(jax.value_and_grad(lbfgs_cost, argnums=1))


def train_step_with_cost_jax(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    train_state: TrainState,
    max_iter: int = 16,
    max_ls: int = 8,
    unroll: bool = True,
) -> tuple[TrainState, utils.TableSol, lbfgs_result]:
    ts = train_state

    # compute
    opt_params = lbfgs.OptParamsLBFGS(
        fun=lbfgs_cost_and_grad,
        max_iter=max_iter,
        max_ls=max_ls,
        tol=1e-5,
        c1=1e-4,
        c2=0.9,
    )
    res = lbfgs.lbfgs(
        opt_params=opt_params,
        x0=train_state.control_flat,
        fun_params=(ts, weights, cost_terms, acc_ref, omega_ref),
        unroll=unroll,
    )

    # convert to non-sim time
    control_sim = utils.Control.from_flat(res[0])
    control = control_sim.refine_control(const.dt_sim, const.dt)
    ctrl_refinement = functools.partial(
        utils.control_refinement, const.dt_sim, const.dt
    )
    acc_ref = ctrl_refinement(acc_ref)
    omega_ref = ctrl_refinement(omega_ref)

    # compute results
    rstate = utils.get_rstate(const.dt, control, ts.rstate0)
    vstate_irl = utils.get_vstate_irl(
        const.vspec_acc,
        const.vspec_omega,
        rstate,
        control,
        ts.control0,
        ts.vstate0_irl,
    )
    vstate_sim = utils.get_vstate(
        const.vspec_acc, const.vspec_omega, acc_ref, omega_ref, ts.vstate0_sim
    )

    # bookkeeping
    table_sol = utils.TableSol(
        x=rstate,
        u=control,
        vstate_irl=vstate_irl,
        vstate_sim=vstate_sim,
        stats=utils.TableStats(
            time=jnp.squeeze(0.0),
            status=jnp.array(0),
            cost=jnp.array(res[1]),
        ),
    )
    next_state = TrainState(
        rstate0=rstate.state[1],
        vstate0_irl=vstate_irl.x_state[0],  # NOT off-by-one
        vstate0_sim=vstate_sim.x_state[1],
        control0=res[0][:6],
        control_flat=res[0],
    )
    return next_state, table_sol, res


train_step_with_cost_jit = jax.jit(
    train_step_with_cost_jax,
    static_argnames=["max_iter", "max_ls", "unroll"],
)


def train_step_with_cost(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    train_state: TrainState,
    **kwargs,
) -> tuple[TrainState, utils.TableSol, lbfgs_result, float]:
    t0 = time.time()
    opt_options = kwargs.get("opt_options", {"maxiter": 16, "maxls": 8})
    res = train_step_with_cost_jit(
        weights,
        cost_terms,
        acc_ref,
        omega_ref,
        train_state,
        opt_options["maxiter"],
        opt_options["maxls"],
    )
    res[0].control0.block_until_ready()
    t1 = time.time()
    return res[0], res[1], res[2], t1 - t0
