"""
We provide optimization components for Stewart platform MPC control.

* Tuning classes: :class:`Weights`, :class:`ExpWeights`, and :class:`CostTerms`.
* The cost function implementation: :func:`cost_flat_jax`.
* A feedback loop for python simulations: :func:`train_step_with_cost`.

The general philosophy is as follows.

* Functions are jax compatible.
* Implementations are hacky (for easy experimentation).
* The number of abstractions should be minimized.

See the :doc:`C++ docs <../cpp>` to see how to integrate the MPC feedback
into a C++ program.
"""

from __future__ import annotations

import dataclasses
import functools
import time
import typing as tp

import numpy as np
import jax
import jax.numpy as jnp

import exp_mpc.stewart_min.robo as robo
import exp_mpc.stewart_min.vest as vest
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
    """Cost function weights.

    Parameters
    ----------
    acc :
        Per-axis weights for linear acceleration: ``[x, y, z]``.
    omega :
        Per-axis weights for angular velocity:
        ``[roll_dot, pitch_dot, yaw_dot]``.
    leg :
        Per-leg weights for leg lengths.
    leg_vel :
        Per-leg weights for leg velocities.
    joint_angle :
        Per-joint weights for top and bottom joint angles.
        Ordering is ``[top_0..top_5, bot_0..bot_5]``.
    roll :
        Roll weight.
    pitch :
        Pitch weight.
    yaw :
        Yaw weight.
    yaw_dot :
        Yaw velocity weight.
    control :
        Per-axis weights for control effort:
        ``[x_dot2, y_dot2, z_dot2, roll_dot2, pitch_dot2, yaw_dot2]``.
    terminal_exp_scale :
        Exponential scaling factor for terminal-state attenuation.
    terminal_vt_scale :
        Global scale for terminal vestibular mismatch term.
    terminal_rt_scale :
        Global scale for terminal robot state mismatch term.
    """

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
    terminal_exp_scale: jax.Array = _init_field(jnp.array(50.0))
    terminal_vt_scale: jax.Array = _init_field(jnp.array(4.0))
    terminal_rt_scale: jax.Array = _init_field(jnp.array(0.2))

    def __post_init__(self) -> None:
        """Validate expected weight-array shapes."""
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
        assert self.terminal_exp_scale.ndim == 0
        assert self.terminal_vt_scale.ndim == 0
        assert self.terminal_rt_scale.ndim == 0

    def _time_scale(self, n: int, name: str) -> jax.Array:
        """Get time scale weights for flat array.

        See the `ExpWeights` class for a nontrivial implementation
        """
        # identity
        return jnp.ones(n, dtype=float)

    def scale_acc(self, n: int) -> jax.Array:
        """Get time expanded weights for acceleration cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            2D array of shape ``(n, 3)`` with per-step and per-axis weights.
        """
        time_scale = self._time_scale(n, "acc")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.acc.size))
        val_scale = jnp.tile(self.acc, (n, 1))
        return time_scale * val_scale

    def scale_omega(self, n: int) -> jax.Array:
        """Get time expanded weights for angular velocity cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            2D array of shape ``(n, 3)`` with per-step and per-axis weights.
        """
        time_scale = self._time_scale(n, "omega")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.omega.size))
        val_scale = jnp.tile(self.omega, (n, 1))
        return time_scale * val_scale

    def scale_leg(self, n: int) -> jax.Array:
        """Get time expanded weights for leg length cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n * 6,)``.
        """
        time_scale = self._time_scale(n, "leg")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.leg.size))
        val_scale = jnp.tile(self.leg, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_leg_vel(self, n: int) -> jax.Array:
        """Get time expanded weights for leg velocity cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n * 6,)``.
        """
        time_scale = self._time_scale(n, "leg_vel")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.leg_vel.size))
        val_scale = jnp.tile(self.leg_vel, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_joint_angle(self, n: int) -> jax.Array:
        """Get time expanded weights for joint angle cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n * 12,)``.
        """
        time_scale = self._time_scale(n, "joint_angle")
        time_scale = jnp.tile(
            time_scale.reshape(-1, 1), (1, self.joint_angle.size)
        )
        val_scale = jnp.tile(self.joint_angle, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_roll(self, n: int) -> jax.Array:
        """Get time expanded weights for roll boundary cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n,)``.
        """
        time_scale = self._time_scale(n, "roll")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.roll.size))
        val_scale = jnp.tile(self.roll, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_pitch(self, n: int) -> jax.Array:
        """Get time expanded weights for pitch boundary cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n,)``.
        """
        time_scale = self._time_scale(n, "pitch")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.pitch.size))
        val_scale = jnp.tile(self.pitch, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_yaw(self, n: int) -> jax.Array:
        """Get time expanded weights for yaw boundary cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n,)``.
        """
        time_scale = self._time_scale(n, "yaw")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.yaw.size))
        val_scale = jnp.tile(self.yaw, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_yaw_dot(self, n: int) -> jax.Array:
        """Get time expanded weights for yaw velocity boundary cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n,)``.
        """
        time_scale = self._time_scale(n, "yaw_dot")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.yaw_dot.size))
        val_scale = jnp.tile(self.yaw_dot, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_control(self, n: int) -> jax.Array:
        """Get time expanded weights for control effort.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n * 6,)``.
        """
        time_scale = self._time_scale(n, "control")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.control.size))
        val_scale = jnp.tile(self.control, (n, 1))
        return jnp.ravel(time_scale * val_scale)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ExpWeights(Weights):
    """Exponential time decaying extension of :class:`Weights`.

    Parameters
    ----------
    alpha_acc :
        Decay rate for accelerations.
    alpha_omega :
        Decay rate for angular velocities.
    alpha_leg :
        Decay rate for leg lengths.
    alpha_leg_vel :
        Decay rate for leg velocities.
    alpha_joint_angle :
        Decay rate for joint angles.
    alpha_roll :
        Decay rate for roll.
    alpha_pitch :
        Decay rate for pitch.
    alpha_yaw :
        Decay rate for yaw.
    alpha_yaw_dot :
        Decay rate for yaw velocities.
    alpha_control :
        Decay rate for control effort.

    Notes
    -----
    The time profile is ``exp(-k / n * alpha)`` where ``k`` is the
    discrete horizon index and ``n`` is horizon length.
    Namely, ``alpha`` is the maximum exponential decrease factor, or
    alternatively, ``alpha`` is the decay rate when time is normalized to unity.
    """
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
    """Container for the boundary penalties used by the MPC objective.

    Parameters
    ----------
    leg_cost :
        Quartic cost for leg length boundary.
    leg_vel_cost :
        Quartic cost for leg velocity boundary.
    joint_angle_cost :
        Quartic cost for joint angle boundary.
    roll_cost :
        Quartic cost for roll boundary.
    pitch_cost :
        Quartic cost for pitch boundary.
    yaw_cost :
        Quartic cost for yaw boundary.
    yaw_dot_cost :
        Quartic cost for yaw rate boundary.
    """
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


def _acc_cost_arr(
    weights: Weights,
    vstate_irl: utils.VState,
    vstate_sim: utils.VState,
) -> jax.Array:
    """Head acceleration cost terms."""
    w = weights.scale_acc(vstate_irl.size)
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
) -> jax.Array:
    """Head acceleration cost."""
    cost_arr = _acc_cost_arr(weights, vstate_irl, vstate_sim)
    return 0.5 * jnp.mean(cost_arr)


def _omega_cost_arr(
    weights: Weights,
    vstate_irl: utils.VState,
    vstate_sim: utils.VState,
) -> jax.Array:
    """Angular velocity cost terms."""
    w = weights.scale_omega(vstate_irl.size)  # or `vstate_sim.size`
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
) -> jax.Array:
    """Angular velocity cost."""
    cost_arr = _omega_cost_arr(weights, vstate_irl, vstate_sim)
    return 0.5 * jnp.mean(cost_arr)


def _leg_boundary_cost_arr(
    robo_geom: robo.RoboGeom,
    weights: Weights,
    cost_terms: CostTerms,
    rstate: utils.RState,
    use_rotary: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Include leg length and leg velocity costs.

    By using automatic differentiation, we can compute the lengths and the
    velocities cheaper than computing them separately, which is productive.
    """
    leg_pos_vel = functools.partial(
        utils.leg_pos_vel,
        robo_geom=robo_geom,
        use_rotary=use_rotary,
    )
    rstate = rstate.pop0()
    lengths, vels = jax.vmap(leg_pos_vel)(rstate)
    lengths = jnp.ravel(lengths)
    vels = jnp.ravel(vels)
    length_costs = jax.vmap(cost_terms.leg_cost)(lengths)
    vel_costs = jax.vmap(cost_terms.leg_vel_cost)(vels)
    w_len = weights.scale_leg(rstate.size)
    w_vel = weights.scale_leg_vel(rstate.size)
    length_cost_arr = jnp.reshape(length_costs * w_len, shape=(-1, 6))
    vel_cost_arr = jnp.reshape(vel_costs * w_vel, shape=(-1, 6))
    return length_cost_arr, vel_cost_arr


def _leg_boundary_cost(
    robo_geom: robo.RoboGeom,
    weights: Weights,
    cost_terms: CostTerms,
    rstate: utils.RState,
    use_rotary: bool,
) -> jax.Array:
    """Include leg length and leg velocity costs.

    By using automatic differentiation, we can compute the lengths and the
    velocities cheaper than computing them separately, which is productive.
    """
    length_cost_arr, vel_cost_arr = _leg_boundary_cost_arr(
        robo_geom, weights, cost_terms, rstate, use_rotary
    )
    length_cost_val = jnp.sum(jnp.mean(length_cost_arr, axis=0))
    vel_cost_val = jnp.sum(jnp.mean(vel_cost_arr, axis=0))
    return length_cost_val + vel_cost_val


def _joint_angles(
    robo_geom: robo.RoboGeom,
    rstate: utils.RState,
    use_rotary: bool,
) -> jax.Array:
    return jnp.concatenate(
        utils.angle_joint(rstate, robo_geom=robo_geom, use_xy=use_rotary)
    )


def _joint_angle_boundary_cost_arr(
    robo_geom: robo.RoboGeom,
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
    use_rotary: bool = True,
) -> jax.Array:
    """Joint angle cost."""
    joint_angles_part = functools.partial(
        _joint_angles,
        robo_geom,
        use_rotary=use_rotary,
    )
    rstate = rstate.pop0()
    angles = jnp.ravel(jax.vmap(joint_angles_part)(rstate))
    costs = jax.vmap(costs.joint_angle_cost)(angles)
    w = weights.scale_joint_angle(rstate.size)
    return (costs * w).reshape(-1, 12)


def _joint_angle_boundary_cost(
    robo_geom: robo.RoboGeom,
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
    use_rotary: bool,
) -> jax.Array:
    """Joint angle cost.

    This is about 3 times more expensive to compute than the other
    cost functions (including boundary cost functions).
    """
    cost_arr = _joint_angle_boundary_cost_arr(
        robo_geom,
        weights,
        costs,
        rstate,
        use_rotary,
    )
    return jnp.sum(jnp.mean(cost_arr, axis=0))


def _roll_boundary_cost_arr(
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
) -> jax.Array:
    rstate = rstate.pop0()
    roll = rstate.roll
    costs = jax.vmap(costs.roll_cost)(roll)
    w = weights.scale_roll(rstate.size)
    return costs * w


def _roll_boundary_cost(
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
) -> jax.Array:
    cost_arr = _roll_boundary_cost_arr(weights, costs, rstate)
    return jnp.mean(cost_arr)


def _pitch_boundary_cost_arr(
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
) -> jax.Array:
    rstate = rstate.pop0()
    pitch = rstate.pitch
    costs = jax.vmap(costs.pitch_cost)(pitch)
    w = weights.scale_pitch(rstate.size)
    return costs * w


def _pitch_boundary_cost(
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
) -> jax.Array:
    cost_arr = _pitch_boundary_cost_arr(weights, costs, rstate)
    return jnp.mean(cost_arr)


def _yaw_boundary_cost_arr(
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
) -> jax.Array:
    rstate = rstate.pop0()
    yaw = rstate.yaw
    costs = jax.vmap(costs.yaw_cost)(yaw)
    w = weights.scale_yaw(rstate.size)
    return costs * w


def _yaw_boundary_cost(
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
) -> jax.Array:
    cost_arr = _yaw_boundary_cost_arr(weights, costs, rstate)
    return jnp.mean(cost_arr)


def _yaw_dot_boundary_cost_arr(
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
) -> jax.Array:
    rstate = rstate.pop0()
    yaw_dot = rstate.yaw_dot
    yaw_dot = jnp.ravel(jnp.transpose(yaw_dot))
    costs = jax.vmap(costs.yaw_dot_cost)(yaw_dot)
    w = weights.scale_yaw_dot(rstate.size)
    return costs * w


def _yaw_dot_boundary_cost(
    weights: Weights,
    costs: CostTerms,
    rstate: utils.RState,
) -> jax.Array:
    cost_arr = _yaw_dot_boundary_cost_arr(weights, costs, rstate)
    return jnp.mean(cost_arr)


def _control_cost_arr(
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
    cost_arr = _control_cost_arr(weights, control)
    return 0.5 * jnp.sum(jnp.mean(cost_arr, axis=0))


def _terminal_cost(
    robo_geom: robo.RoboGeom,
    vspec_acc: vest.VSpec,
    vspec_omega: vest.VSpec,
    weights: Weights,
    rstate: utils.RState,
    vstate_irl: utils.VState,
    vstate_sim: utils.VState,
) -> jax.Array:
    # setup
    vi = vstate_irl
    vs = vstate_sim

    a_P = vspec_acc.P
    o_P = vspec_omega.P

    a_num = vspec_acc.n_state
    o_num = vspec_omega.n_state
    idx = np.cumsum([0] + [a_num] * 3 + [o_num] * 3)

    x_accx0 = a_P @ vs.x_state[0, idx[0] : idx[1]]
    x_accx1 = a_P @ vs.x_state[-1, idx[0] : idx[1]]
    x_accy0 = a_P @ vs.x_state[0, idx[1] : idx[2]]
    x_accy1 = a_P @ vs.x_state[-1, idx[1] : idx[2]]
    x_accz0 = a_P @ vs.x_state[0, idx[2] : idx[3]]
    x_accz1 = a_P @ vs.x_state[-1, idx[2] : idx[3]]
    x_omegax0 = o_P @ vs.x_state[0, idx[3] : idx[4]]
    x_omegax1 = o_P @ vs.x_state[-1, idx[3] : idx[4]]
    x_omegay0 = o_P @ vs.x_state[0, idx[4] : idx[5]]
    x_omegay1 = o_P @ vs.x_state[-1, idx[4] : idx[5]]
    x_omegaz0 = o_P @ vs.x_state[0, idx[5] : idx[6]]
    x_omegaz1 = o_P @ vs.x_state[-1, idx[5] : idx[6]]

    irl_x_omegaz1 = o_P @ vi.x_state[-1, idx[5] : idx[6]]
    o_diff = irl_x_omegaz1 - x_omegaz1

    def o_V(x):
        return jnp.dot(vspec_omega.V @ x, x)

    # compute

    def scale(x):
        return jnp.exp(-weights.terminal_exp_scale * jnp.sum(jnp.square(x)))

    vt_cost = o_V(o_diff) * (scale(x_omegaz0) * scale(x_omegaz1))
    vt_cost *= weights.terminal_vt_scale

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

    scales = scale0 * scale1 * weights.terminal_rt_scale
    last_state = rstate.state[-1]
    last_state = last_state.at[:3].subtract(robo_geom.cart_home)
    rt_cost = jnp.sum(jnp.square(last_state) * scales)
    rt_cost += jnp.square(rstate.state[-1][5]) * (
        scale(x_omegaz0) * scale(x_omegaz1) * weights.terminal_rt_scale * 1e1
    )

    return rt_cost + vt_cost


def _cost(
    control: utils.Control,
    rstate0: jax.Array,
    control0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    weights: Weights,
    cost_terms: CostTerms,
    dt: jax.Array,
    robo_geom: robo.RoboGeom,  # static
    vspec_acc: vest.VSpec,  # static
    vspec_omega: vest.VSpec,  # static
    use_rotary: bool = True,  # static
    use_terminal: bool = True,  # static
) -> jax.Array:
    # precompute states
    rstate, vstate_irl, vstate_sim = utils.get_states_with_eigen(
        dt,
        vspec_acc,
        vspec_omega,
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
    cost += _acc_cost(weights, vstate_irl, vstate_sim)
    cost += _omega_cost(weights, vstate_irl, vstate_sim)
    cost += _leg_boundary_cost(
        robo_geom,
        weights,
        cost_terms,
        rstate,
        use_rotary,
    )
    cost += _joint_angle_boundary_cost(
        robo_geom,
        weights,
        cost_terms,
        rstate,
        use_rotary,
    )
    cost += _roll_boundary_cost(weights, cost_terms, rstate)
    cost += _pitch_boundary_cost(weights, cost_terms, rstate)
    cost += _yaw_boundary_cost(weights, cost_terms, rstate)
    cost += _yaw_dot_boundary_cost(weights, cost_terms, rstate)
    cost += _control_cost(weights, control)
    if use_terminal:
        cost += _terminal_cost(
            robo_geom,
            vspec_acc,
            vspec_omega,
            weights,
            rstate,
            vstate_irl,
            vstate_sim,
        )
    return cost


#######################
# scipy cost wrappers #
#######################

_cost_static_argnames = [
    "robo_geom",
    "vspec_acc",
    "vspec_omega",
    "use_rotary",
    "use_terminal",
]

@functools.partial(
    jax.jit,
    static_argnames=_cost_static_argnames,
)
def cost_flat_jax(
    control_flat: jax.Array,
    rstate0: jax.Array,
    control0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    weights: Weights,
    cost_terms: CostTerms,
    dt: jax.Array,
    robo_geom: robo.RoboGeom,
    vspec_acc: vest.VSpec,
    vspec_omega: vest.VSpec,
    use_rotary: bool = True,
    use_terminal: bool = True,
) -> jax.Array:
    """Evaluate MPC objective from a flattened control trajectory.

    This flattening wrapper is easily motivated.
    Optimization algorithms prefer that the optimization variables are
    represented as a vector, i.e., a flat array.

    Parameters
    ----------
    control_flat :
        Flattened control sequence with ordering
        ``[x, y, z, roll, pitch, yaw]`` per time step.
    rstate0 :
        Current robot state.
    control0 :
        Current robot accelerations.
        (Last robot control.)
    vstate0_irl :
        Initial vestibular state for the in-real-life person.
    vstate0_sim :
        Initial vestibular state for the simulated/reference person.
    acc_ref :
        Reference linear acceleration trajectory in the head frame.
    omega_ref :
        Reference angular velocity trajectory in the head frame.
    weights :
        Cost weights.
    cost_terms :
        Quartic boundary penalties.
    dt :
        Time step.
    robo_geom :
        Stewart platform geometry.
    vspec_acc :
        Vestibular acceleration model specification.
    vspec_omega :
        Vestibular angular velocity model specification.
    use_rotary :
        ``True`` to simulate a rotary platform on top of the Stewart platform.
        ``False`` for no rotary platform.
    use_terminal :
        ``True`` to include the terminal cost.
        The terminal cost is mainly used for smooth tracking to home.

    Returns
    -------
    cost :
        Scalar MPC objective value.
    """
    control = utils.Control.from_flat(control_flat)
    return _cost(
        control=control,
        rstate0=rstate0,
        control0=control0,
        vstate0_irl=vstate0_irl,
        vstate0_sim=vstate0_sim,
        acc_ref=acc_ref,
        omega_ref=omega_ref,
        weights=weights,
        cost_terms=cost_terms,
        vspec_acc=vspec_acc,
        vspec_omega=vspec_omega,
        dt=dt,
        robo_geom=robo_geom,
        use_rotary=use_rotary,
        use_terminal=use_terminal,
    )


@functools.partial(
    jax.jit,
    static_argnames=_cost_static_argnames,
)
def cost_and_grad_flat_jax(
    control_flat: jax.Array,
    rstate0: jax.Array,
    control0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    weights: Weights,
    cost_terms: CostTerms,
    dt: jax.Array,
    robo_geom: robo.RoboGeom,
    vspec_acc: vest.VSpec,
    vspec_omega: vest.VSpec,
    use_rotary: bool = True,
    use_terminal: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate MPC objective and gradient for flattened controls.

    Parameters
    ----------
    control_flat :
        Flattened control sequence with ordering
        ``[x, y, z, roll, pitch, yaw]`` per time step.
    rstate0 :
        Current robot state.
    control0 :
        Current robot accelerations.
        (Last robot control.)
    vstate0_irl :
        Initial vestibular state for the in-real-life person.
    vstate0_sim :
        Initial vestibular state for the simulated/reference person.
    acc_ref :
        Reference linear acceleration trajectory in the head frame.
    omega_ref :
        Reference angular velocity trajectory in the head frame.
    weights :
        Cost weights.
    cost_terms :
        Quartic boundary penalties.
    dt :
        Time step.
    robo_geom :
        Stewart platform geometry.
    vspec_acc :
        Vestibular acceleration model specification.
    vspec_omega :
        Vestibular angular velocity model specification.
    use_rotary :
        ``True`` to simulate a rotary platform on top of the Stewart platform.
        ``False`` for no rotary platform.
    use_terminal :
        ``True`` to include the terminal cost.
        The terminal cost is mainly used for smooth tracking to home.

    Returns
    -------
    cost :
        Scalar MPC cost.
    grad :
        Control (flat) gradient of MPC cost.
    """
    cost_and_grad = jax.value_and_grad(cost_flat_jax, argnums=0)
    return cost_and_grad(
        control_flat,
        rstate0=rstate0,
        control0=control0,
        vstate0_irl=vstate0_irl,
        vstate0_sim=vstate0_sim,
        acc_ref=acc_ref,
        omega_ref=omega_ref,
        weights=weights,
        cost_terms=cost_terms,
        vspec_acc=vspec_acc,
        vspec_omega=vspec_omega,
        dt=dt,
        robo_geom=robo_geom,
        use_rotary=use_rotary,
        use_terminal=use_terminal,
    )


################
# jax training #
################


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TrainState:
    """State for training.

    The name was motivated by the machine learning community.
    See :class:`flax.training.train_state.TrainState`.
    Essentially, the container includes the updated parameters after each MPC
    optimization iteration.

    Parameters
    ----------
    rstate0 :
        Current robot state.
    vstate0_irl :
        Current vestibular state for the in-real-life person.
    vstate0_sim :
        Current vestibular state for the simulated/reference person.
    control0 :
        Current robot acceleration.
    control_flat :
        Flattened control sequence for the MPC horizon.
        (Last optimization solution.)
    """

    rstate0: jax.Array
    vstate0_irl: jax.Array
    vstate0_sim: jax.Array
    control0: jax.Array
    control_flat: jax.Array

    @classmethod
    def zero_init(
        cls,
        robo_geom: robo.RoboGeom,
        horizon_num: int,
        vspec_acc: vest.VSpec,
        vspec_omega: vest.VSpec,
        vstate0_mode: tp.Optional[tuple[str, str]] = None,
    ) -> "TrainState":
        """Init train state with zeros.

        Parameters
        ----------
        robo_geom :
            Stewart platform geometry.
            Used for home positioning.
        horizon_num :
            Horizon length.
        vspec_acc :
            Vestibular specification for acceleration.
        vspec_omega :
            Vestibular specification for angular velocity.
        vstate0_mode :
            Determines if the initial vestibular states should be initialized
            to respect gravity.
            The first and second entries represent the irl and sim conditions,
            respectively.
            If not ``None``, the options are ``"earth"`` or ``"moon"``.

        Returns
        -------
        train_state :
            Zeroed train state.
        """
        acc_num = vspec_acc.E0.shape[0]
        omega_num = vspec_omega.E0.shape[0]
        v0_earth = vspec_acc.v0_earth
        v0_moon = vspec_acc.v0_moon
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

        rstate0 = zeros(r_num)
        rstate0 = rstate0.at[:3].add(robo_geom.cart_home)

        return cls(
            rstate0=rstate0,
            vstate0_irl=vstate0_irl,
            vstate0_sim=vstate0_sim,
            control0=zeros(u_num),
            control_flat=zeros(u_num * horizon_num),
        )


def lbfgs_cost(
    weights: Weights,
    cost_terms: CostTerms,
    dt: jax.Array,
    vspec_acc: vest.VSpec,
    vspec_omega: vest.VSpec,
    robo_geom: robo.RoboGeom,
    use_terminal: bool,
    args: tuple[TrainState, jax.Array, jax.Array],
    control_flat: jax.Array,
) -> jax.Array:
    """L-BFGS wrapper of :func:`cost_flat_jax`.

    Parameters
    ----------
    weights :
        Cost weights.
    cost_terms :
        Quartic boundary penalties.
    dt :
        Time step used by the MPC horizon.
    vspec_acc :
        Vestibular acceleration model specification.
    vspec_omega :
        Vestibular angular velocity model specification.
    robo_geom :
        Stewart platform geometry.
    use_terminal :
        Whether terminal costs are enabled.
    args :
        Tuple ``(train_state, acc_ref, omega_ref)`` passed through L-BFGS.
        These are the arguments that change during each MPC control cycle.
    control_flat :
        Flattened control sequence being optimized.

    Returns
    -------
    cost :
        Scalar MPC objective value.
    """
    train_state, acc_ref, omega_ref = args
    return cost_flat_jax(
        control_flat=control_flat,
        rstate0=train_state.rstate0,
        control0=train_state.control0,
        vstate0_irl=train_state.vstate0_irl,
        vstate0_sim=train_state.vstate0_sim,
        acc_ref=acc_ref,
        omega_ref=omega_ref,
        weights=weights,
        cost_terms=cost_terms,
        vspec_acc=vspec_acc,
        vspec_omega=vspec_omega,
        dt=dt,
        robo_geom=robo_geom,
        use_rotary=True,
        use_terminal=use_terminal,
    )


lbfgs_cost_and_grad = jax.jit(
    jax.value_and_grad(lbfgs_cost, argnums=-1),
    static_argnames=["vspec_acc", "vspec_omega", "robo_geom", "use_terminal"],
)


def train_step_with_cost_jax(
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    train_state: TrainState,
    weights: Weights,
    cost_terms: CostTerms,
    dt: jax.Array,
    dt_mpc: jax.Array,
    robo_geom: robo.RoboGeom,
    vspec_acc: vest.VSpec,
    vspec_omega: vest.VSpec,
    vspec_acc_mpc: vest.VSpec,
    vspec_omega_mpc: vest.VSpec,
    max_iter: int = 16,
    max_ls: int = 8,
    unroll: bool = False,
    use_terminal: bool = True,
) -> tuple[TrainState, utils.TableSol, lbfgs_result]:
    """Run one MPC control cycle with JAX L-BFGS.

    Parameters
    ----------
    acc_ref :
        Reference linear acceleration trajectory.
        Shape: ``(horizon_num, 3)``.
    omega_ref :
        Reference angular velocity trajectory.
        Shape: ``(horizon_num, 3)``.
    train_state :
        Current MPC state.
    weights :
        Cost weights.
    cost_terms :
        Quartic boundary penalties.
    dt :
        Robot control cycle.
    dt_mpc :
        Optimization horizon integration step.
        We require ``dt_mpc >= dt``.
    robo_geom :
        Stewart platform geometry.
    vspec_acc :
        Vestibular acceleration model, with integration time step ``dt``.
    vspec_omega :
        Vestibular angular velocity model, with integration time step ``dt``.
    vspec_acc_mpc :
        Vestibular acceleration model, with integration time step ``dt_mpc``.
    vspec_omega_mpc :
        Vestibular angular velocity model, with integration time step
        ``dt_mpc``.
    max_iter :
        Maximum L-BFGS iterations.
    max_ls :
        Maximum line search iterations per L-BFGS step.
    unroll :
        Whether to unroll L-BFGS loop (JAX control-flow choice).
    use_terminal :
        Whether to include terminal penalties.

    Returns
    -------
    next_state :
        Updated MPC state.
    table_sol :
        MPC solution trajectory and statistics.
    lbfgs_res :
        L-BFGS optimizer tuple ``(minimizer, value, gradient)``.
    """
    ts = train_state

    # compute
    opt_fun = functools.partial(
        lbfgs_cost_and_grad,
        weights,
        cost_terms,
        dt_mpc,
        vspec_acc_mpc,
        vspec_omega_mpc,
        robo_geom,
        use_terminal,
    )
    opt_params = lbfgs.OptParamsLBFGS(
        fun=opt_fun,
        max_iter=max_iter,
        max_ls=max_ls,
        tol=1e-5,
        c1=1e-4,
        c2=0.9,
    )
    res = lbfgs.lbfgs(
        opt_params=opt_params,
        x0=train_state.control_flat,
        fun_params=(ts, acc_ref, omega_ref),
        unroll=unroll,
    )

    # convert to non-sim time
    control_sim = utils.Control.from_flat(res[0])
    control = control_sim.refine_control(dt_mpc, dt)
    ctrl_refinement = functools.partial(utils.control_refinement, dt_mpc, dt)
    acc_ref = ctrl_refinement(acc_ref)
    omega_ref = ctrl_refinement(omega_ref)

    # compute results
    rstate = utils.get_rstate(dt, control, ts.rstate0)
    vstate_irl = utils.get_vstate_irl(
        vspec_acc,
        vspec_omega,
        rstate,
        control,
        ts.control0,
        ts.vstate0_irl,
    )
    vstate_sim = utils.get_vstate(
        vspec_acc, vspec_omega, acc_ref, omega_ref, ts.vstate0_sim
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
    static_argnames=[
        "dt",
        "dt_mpc",
        "vspec_acc",
        "vspec_omega",
        "vspec_acc_mpc",
        "vspec_omega_mpc",
        "robo_geom",
        "max_iter",
        "max_ls",
        "unroll",
        "use_terminal",
    ],
)


def train_step_with_cost(
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    train_state: TrainState,
    weights: Weights,
    cost_terms: CostTerms,
    dt: jax.Array,
    dt_mpc: jax.Array,
    robo_geom: robo.RoboGeom,
    vspec_acc: vest.VSpec,
    vspec_omega: vest.VSpec,
    vspec_acc_mpc: vest.VSpec,
    vspec_omega_mpc: vest.VSpec,
    max_iter=16,
    max_ls=8,
    unroll: bool = False,
    use_terminal: bool = True,
) -> tuple[TrainState, utils.TableSol, lbfgs_result, float]:
    """Run one MPC control cycle with JAX L-BFGS, and measure wall time.

    Parameters
    ----------
    acc_ref :
        Reference linear acceleration trajectory.
        Shape: ``(horizon_num, 3)``.
    omega_ref :
        Reference angular velocity trajectory.
        Shape: ``(horizon_num, 3)``.
    train_state :
        Current MPC state.
    weights :
        Cost weights.
    cost_terms :
        Quartic boundary penalties.
    dt :
        Robot control cycle.
    dt_mpc :
        Optimization horizon integration step.
        We require ``dt_mpc >= dt``.
    robo_geom :
        Stewart platform geometry.
    vspec_acc :
        Vestibular acceleration model, with integration time step ``dt``.
    vspec_omega :
        Vestibular angular velocity model, with integration time step ``dt``.
    vspec_acc_mpc :
        Vestibular acceleration model, with integration time step ``dt_mpc``.
    vspec_omega_mpc :
        Vestibular angular velocity model, with integration time step
        ``dt_mpc``.
    max_iter :
        Maximum L-BFGS iterations.
    max_ls :
        Maximum line search iterations per L-BFGS step.
    unroll :
        Whether to unroll L-BFGS loop (JAX control-flow choice).
    use_terminal :
        Whether to include terminal penalties.

    Returns
    -------
    next_state :
        Updated MPC state.
    table_sol :
        MPC solution trajectory and statistics.
    lbfgs_res :
        L-BFGS optimizer tuple ``(minimizer, value, gradient)``.
    elapsed_time :
        Wall-time in seconds for calling the jit-ed
        :func:`train_step_with_cost_jax`.
    """
    t0 = time.time()
    res = train_step_with_cost_jit(
        weights=weights,
        cost_terms=cost_terms,
        acc_ref=acc_ref,
        omega_ref=omega_ref,
        train_state=train_state,
        dt=dt,
        dt_mpc=dt_mpc,
        vspec_acc=vspec_acc,
        vspec_omega=vspec_omega,
        vspec_acc_mpc=vspec_acc_mpc,
        vspec_omega_mpc=vspec_omega_mpc,
        robo_geom=robo_geom,
        max_iter=max_iter,
        max_ls=max_ls,
        unroll=unroll,
        use_terminal=use_terminal,
    )
    res[0].control0.block_until_ready()
    t1 = time.time()
    return res[0], res[1], res[2], t1 - t0
