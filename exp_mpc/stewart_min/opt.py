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
import typing as tp

import numpy as np
import jax
import jax.numpy as jnp

import exp_mpc.stewart_min.const as const
import exp_mpc.stewart_min.utils as utils
import exp_mpc.stewart_min.quartic_cost as quartic_cost

# make sure to enable 64-bit precision for jax
# this is necessary for good performance
# use the following line when importing this library
# `jax.config.update("jax_enable_x64", True)`


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
    yaw: jax.Array = _init_field(jnp.ones(1))
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
        assert self.yaw.ndim == 1
        assert self.yaw.shape[0] == 1
        assert self.control.ndim == 1
        assert self.control.shape[0] == 6

        assert jnp.issubdtype(self.acc.dtype, jnp.floating)
        assert jnp.issubdtype(self.omega.dtype, jnp.floating)
        assert jnp.issubdtype(self.leg.dtype, jnp.floating)
        assert jnp.issubdtype(self.leg_vel.dtype, jnp.floating)
        assert jnp.issubdtype(self.joint_angle.dtype, jnp.floating)
        assert jnp.issubdtype(self.yaw.dtype, jnp.floating)
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

    def scale_yaw(self, n: int) -> jax.Array:
        """Get scale weights for flat yaw angle array."""
        time_scale = self._time_scale(n, "yaw")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.yaw.size))
        val_scale = jnp.tile(self.yaw, (n, 1))
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
    alpha_yaw: jax.Array = _init_field(jnp.ones(1) * 0.0)
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
            "yaw": self.alpha_yaw,
            "control": self.alpha_control,
        }
        return jnp.exp(-jnp.arange(n, dtype=float) / n * alpha_map[name])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CostTerms:
    leg_cost: quartic_cost.QuarticCost
    leg_vel_cost: quartic_cost.QuarticCost
    joint_angle_cost: quartic_cost.QuarticCost
    yaw_cost: quartic_cost.QuarticCost


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


@jax.vmap
def _v_acc(v, u):
    """Update vestibular linear accelerations, discretely."""
    return const.E0_acc @ v + np.squeeze(const.E1_acc) * u


def _head_xyz_acc_cost_single(
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    ref: jax.Array,
    rstate: utils.RState,
    control: utils.Control,
    w: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Cost for a single input pairing."""
    # robot
    R = utils.rot(rstate)
    acc = jnp.array([control.x, control.y, control.z])
    world = acc + const.gravity
    head = R.T @ world

    # vestibular
    vstate1_irl = _v_acc(vstate0_irl.reshape(3, -1), head)
    vstate1_sim = _v_acc(vstate0_sim.reshape(3, -1), ref)

    # cost
    diff = jnp.squeeze(const.C_acc @ (vstate1_irl - vstate1_sim).T)
    diff *= w
    delta_xy = diff.at[:2].get()
    delta_z = diff.at[2].get()
    cost = _hyper(delta_xy @ delta_xy) + _hyper(delta_z * delta_z)

    return jnp.ravel(vstate1_irl), jnp.ravel(vstate1_sim), cost


def _head_xyz_acc_cost_scan(
    carry: tuple[jax.Array, jax.Array],
    args: tuple[jax.Array, utils.RState, utils.Control, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    vstate1_irl, vstate1_sim, cost = _head_xyz_acc_cost_single(
        carry[0], carry[1], args[0], args[1], args[2], args[3]
    )
    return (vstate1_irl, vstate1_sim), cost


def head_xyz_acc_cost_arr(
    weights: Weights,
    acc_ref: jax.Array,
    rstate: utils.RState,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    control0: jax.Array,
    control: utils.Control,
) -> jax.Array:
    """Head acceleration cost terms."""
    # assert
    assert acc_ref.ndim == 2
    assert acc_ref.shape[1] == 3
    assert acc_ref.shape[0] == control.x.size
    assert control0.shape == (3,)

    # setup
    w = weights.scale_acc(control.size)
    assert w.shape == acc_ref.shape

    # update vstate0_irl (for closed-loop feedback)
    R = utils.rot(rstate.get0())
    u0 = R.T @ (control0 + const.gravity)
    vstate0_irl = jnp.ravel(_v_acc(vstate0_irl.reshape(3, -1), u0))

    # skip initial conditions in state
    _, cost_arr = jax.lax.scan(
        f=_head_xyz_acc_cost_scan,
        init=(vstate0_irl, vstate0_sim),
        xs=(acc_ref, rstate.pop0(), control, w),
    )
    return cost_arr


def _head_xyz_acc_cost(
    weights: Weights,
    acc_ref: jax.Array,
    rstate: utils.RState,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    control0: jax.Array,
    control: utils.Control,
) -> jax.Array:
    """Head acceleration cost."""
    cost_arr = head_xyz_acc_cost_arr(
        weights, acc_ref, rstate, vstate0_irl, vstate0_sim, control0, control
    )
    return 0.5 * jnp.sum(jnp.mean(cost_arr, axis=0))


@jax.vmap
def _v_omega(v, u):
    """Update vestibular angular velocity, discretely."""
    return const.E0_omega @ v + np.squeeze(const.E1_omega) * u


def _omega_cost_single(
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    ref: jax.Array,
    rstate: utils.RState,
    w: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Cost for a single input pairing."""
    u = utils.angle_vel(rstate)
    vstate1_irl = _v_omega(vstate0_irl.reshape(3, -1), u)
    vstate1_sim = _v_omega(vstate0_sim.reshape(3, -1), ref)
    diff = jnp.squeeze(const.C_omega @ (vstate1_irl - vstate1_sim).T)
    diff *= w
    cost = _hyper(diff @ diff)
    return jnp.ravel(vstate1_irl), jnp.ravel(vstate1_sim), cost


def _omega_cost_scan(
    carry: tuple[jax.Array, jax.Array],
    args: tuple[jax.Array, utils.RState, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    vstate1_irl, vstate1_sim, cost = _omega_cost_single(
        carry[0], carry[1], args[0], args[1], args[2]
    )
    return (vstate1_irl, vstate1_sim), cost


def omega_cost_arr(
    weights: Weights,
    omega_ref: jax.Array,
    rstate: utils.RState,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
) -> jax.Array:
    """Angular velocity cost."""
    assert omega_ref.ndim == 2
    assert omega_ref.shape[1] == 3
    assert omega_ref.shape[0] == rstate.roll.size - 1

    # update vstate0_irl (for closed-loop feedback)
    w = weights.scale_omega(rstate.roll.size - 1)
    u0 = utils.angle_vel(rstate.get0())
    vstate0_irl = jnp.ravel(_v_omega(vstate0_irl.reshape(3, -1), u0))

    # skip initial conditions in state
    _, cost_arr = jax.lax.scan(
        f=_omega_cost_scan,
        init=(vstate0_irl, vstate0_sim),
        xs=(omega_ref, rstate.pop0(), w),
    )
    return cost_arr


def _omega_cost(
    weights: Weights,
    omega_ref: jax.Array,
    rstate: utils.RState,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
) -> jax.Array:
    """Angular velocity cost."""
    cost_arr = omega_cost_arr(
        weights, omega_ref, rstate, vstate0_irl, vstate0_sim
    )
    return 0.5 * jnp.sum(jnp.mean(cost_arr, axis=0))


def leg_boundary_cost_arr(
    weights: Weights,
    length_cost: quartic_cost.QuarticCost,
    vel_cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> tuple[jax.Array, jax.Array]:
    """Include leg length and leg velocity costs.

    By using automatic differentiation, we can compute the lengths and the
    velocities cheaper than computing them separately, which is productive.
    """
    lengths, vels = jax.vmap(utils.leg_pos_vel)(state.pop0())
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
) -> jax.Array:
    """Include leg length and leg velocity costs.

    By using automatic differentiation, we can compute the lengths and the
    velocities cheaper than computing them separately, which is productive.
    """
    length_cost_arr, vel_cost_arr = leg_boundary_cost_arr(
        weights, length_cost, vel_cost, state, control
    )
    length_cost_val = jnp.sum(jnp.mean(length_cost_arr, axis=0))
    vel_cost_val = jnp.sum(jnp.mean(vel_cost_arr, axis=0))
    return length_cost_val + vel_cost_val


def joint_angles(state: utils.RState) -> jax.Array:
    return jnp.concatenate(utils.angle_joint(state))


def joint_angle_boundary_cost_arr(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    """Joint angle cost.

    This is about 3 times more expensive to compute than the other
    cost functions (including boundary cost functions).
    """
    angles = jnp.ravel(jax.vmap(joint_angles)(state.pop0()))
    costs = jax.vmap(cost)(angles)
    w = weights.scale_joint_angle(control.size)
    return (costs * w).reshape(-1, 12)


def _joint_angle_boundary_cost(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: utils.RState,
    control: utils.Control,
) -> jax.Array:
    """Joint angle cost.

    This is about 3 times more expensive to compute than the other
    cost functions (including boundary cost functions).
    """
    cost_arr = joint_angle_boundary_cost_arr(weights, cost, state, control)
    return jnp.sum(jnp.mean(cost_arr, axis=0))


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
    return jnp.sum(jnp.mean(cost_arr, axis=0))


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
    rstate: tp.Optional[utils.RState] = None,  # should be None in general
) -> jax.Array:
    # precompute
    if rstate is None:
        rstate = utils.get_rstate(control, rstate0)

    # partition vstates into helpful components
    vstate_acc_num = 3 * const.E0_acc.shape[0]
    vstate0_irl_acc = vstate0_irl[:vstate_acc_num]
    vstate0_irl_omega = vstate0_irl[vstate_acc_num:]
    vstate0_sim_acc = vstate0_sim[:vstate_acc_num]
    vstate0_sim_omega = vstate0_sim[vstate_acc_num:]

    # cost
    cost = jnp.array(0.0)
    cost += _head_xyz_acc_cost(
        weights,
        acc_ref,
        rstate,
        vstate0_irl_acc,
        vstate0_sim_acc,
        control0[:3],
        control,
    )
    cost += _omega_cost(
        weights, omega_ref, rstate, vstate0_irl_omega, vstate0_sim_omega
    )
    cost += _leg_boundary_cost(
        weights, cost_terms.leg_cost, cost_terms.leg_vel_cost, rstate, control
    )
    cost += _joint_angle_boundary_cost(
        weights, cost_terms.joint_angle_cost, rstate, control
    )
    cost += _yaw_boundary_cost(weights, cost_terms.yaw_cost, rstate, control)
    cost += _control_cost(weights, control)
    return cost


#######################
# scipy cost wrappers #
#######################


@jax.jit
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
    )


@jax.jit
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
        Previous staet
    vstate0_sim :

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

    cost_and_grad = functools.partial(cost_and_grad_flat_jax, *params)

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
    def zero_init(cls, horizon_num: int) -> "TrainState":
        """Init train state with zeros.

        Parameters
        ----------
        horizon_num :
            Number of time steps in horizon length.

        Returns
        -------
        Zeroed train state.
        """
        u_num = 6
        r_num = u_num * 2
        v_num = 3 * const.E0_acc.shape[0] + 3 * const.E0_omega.shape[0]

        def zeros(n):
            return jnp.zeros(n, dtype=float)

        return cls(
            rstate0=zeros(r_num),
            vstate0_irl=zeros(v_num),
            vstate0_sim=zeros(v_num),
            control0=zeros(u_num),
            control_flat=zeros(u_num * horizon_num),
        )


def get_scipy_cost_ts(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    train_state: TrainState,
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
    )
