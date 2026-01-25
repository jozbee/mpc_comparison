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


###############
# state model #
###############


@jax.jit
def _discrete_1d_euler(
    x0: jax.Array,
    v0: jax.Array,
    a: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Discrete 1D Euler integration, with scalar initial data.

    Parameters
    ----------
    x0 :
        Initial position.
    v0 :
        Initial velocity.
    a :
        Constant accelerations.

    Returns
    -------
    Integrated position and velocity.
    """
    a = jnp.ravel(a)  # really, an assertion...
    v = jnp.cumsum(jnp.concatenate([jnp.array([v0]), const.dt * a]))
    x = jnp.cumsum(jnp.concatenate([jnp.array([x0]), const.dt * v[1:]]))
    return x, v


###############
# dataclasses #
###############


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class State:
    """Helper for indexing a state array.

    The `state` attribute should be 2D, with rows representing time and columns
    representing state values.
    See the various `property` decorators for specific indexing conventions.

    Note that the initial state is usually stored, so we have a special method
    to ignore it, cf. `pop0`.
    """

    state: jax.Array

    def __post_init__(self):
        if len(self.state.shape) == 2:
            assert self.state.shape[1] == 12
        elif len(self.state.shape) == 1:
            assert self.state.size == 12
        else:
            raise RuntimeError(f"bad state shape: {self.state.shape}")

    @property
    def x(self) -> jax.Array:
        return self.state[..., 0]

    @property
    def y(self) -> jax.Array:
        return self.state[..., 1]

    @property
    def z(self) -> jax.Array:
        return self.state[..., 2]

    @property
    def roll(self) -> jax.Array:
        return self.state[..., 3]

    @property
    def pitch(self) -> jax.Array:
        return self.state[..., 4]

    @property
    def yaw(self) -> jax.Array:
        return self.state[..., 5]

    @property
    def x_dot(self) -> jax.Array:
        return self.state[..., 6]

    @property
    def y_dot(self) -> jax.Array:
        return self.state[..., 7]

    @property
    def z_dot(self) -> jax.Array:
        return self.state[..., 8]

    @property
    def roll_dot(self) -> jax.Array:
        return self.state[..., 9]

    @property
    def pitch_dot(self) -> jax.Array:
        return self.state[..., 10]

    @property
    def yaw_dot(self) -> jax.Array:
        return self.state[..., 11]

    def pop0(self) -> "State":
        """Create a new State without the first time step.

        This is useful when the initial state should be ignored in a
        computation.
        """
        assert len(self.state.shape) == 2
        assert self.state.shape[0] >= 2
        return State(self.state[1:])

    def flatten(self) -> jax.Array:
        return jnp.ravel(self.state)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Control:
    """Helper for indexing a control array.

    The `control` attribute should be 2D, with rows representing time and columns
    representing state values.
    See the various `property` decorators for specific indexing conventions.
    """

    control: jax.Array

    def __post_init__(self):
        if len(self.control.shape) == 2:
            assert self.control.shape[1] == 6
        elif len(self.control.shape) == 1:
            assert self.control.size == 6
        else:
            raise RuntimeError(f"bad control shape: {self.control.shape}")

    @property
    def x(self) -> jax.Array:
        return self.control[..., 0]

    @property
    def y(self) -> jax.Array:
        return self.control[..., 1]

    @property
    def z(self) -> jax.Array:
        return self.control[..., 2]

    @property
    def roll(self) -> jax.Array:
        return self.control[..., 3]

    @property
    def pitch(self) -> jax.Array:
        return self.control[..., 4]

    @property
    def yaw(self) -> jax.Array:
        return self.control[..., 5]

    @property
    def size(self) -> int:
        """Get number of time steps that control represents."""
        if len(self.control.shape) == 2:
            return self.control.shape[0]
        else:
            return 1  # 1 time step

    @classmethod
    def from_flat(cls, flat: jax.Array) -> "Control":
        """Convert a flat array to a control dataclass.

        We assume that the flat array is of the form:
            [x0, y0, z0, roll0, pitch0, yaw0,
             x1, y1, z1, roll1, pitch1, yaw1,
             ...]
        where the first three elements are the x, y and z coordinates of the
        first control point, the next three are the roll, pitch and yaw
        angles of the first control point, and so on for all control points.
        """
        assert flat.size % 6 == 0
        control = jnp.reshape(flat, (-1, 6))
        return cls(control)

    def flatten(self) -> jax.Array:
        return jnp.ravel(self.control)


def get_state(
    control: Control,
    state0: jax.Array,
) -> State:
    """Convert a control dataclass to a state dataclass."""
    state0 = jnp.ravel(state0)
    assert state0.size == 12
    x0, y0, z0, roll0, pitch0, yaw0 = state0[:6]
    x_dot0, y_dot0, z_dot0, roll_dot0, pitch_dot0, yaw_dot0 = state0[6:]

    x, x_dot = _discrete_1d_euler(x0, x_dot0, control.x)
    y, y_dot = _discrete_1d_euler(y0, y_dot0, control.y)
    z, z_dot = _discrete_1d_euler(z0, z_dot0, control.z)
    roll, roll_dot = _discrete_1d_euler(roll0, roll_dot0, control.roll)
    pitch, pitch_dot = _discrete_1d_euler(pitch0, pitch_dot0, control.pitch)
    yaw, yaw_dot = _discrete_1d_euler(yaw0, yaw_dot0, control.yaw)

    non_dots = [x, y, z, roll, pitch, yaw]
    dots = [x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot]
    state = jnp.transpose(jnp.vstack(non_dots + dots))
    return State(state)


###################
# helper wrappers #
###################


def _R(state: State) -> jax.Array:
    """Get the rotation matrix from the state."""
    return utils._get_R(
        phi=state.roll,
        theta=state.pitch,
        psi=state.yaw,
    )


def _R_T(state: State) -> jax.Array:
    """Get the rotation matrix from the state."""
    return jnp.transpose(_R(state))


def _R_dot(state: State) -> jax.Array:
    """Rotation matrix derivative from the state."""
    return utils._get_R_dot(
        phi=state.roll,
        theta=state.pitch,
        psi=state.yaw,
        phi_dot=state.roll_dot,
        theta_dot=state.pitch_dot,
        psi_dot=state.yaw_dot,
    )


def length_and_vel(state: State) -> tuple[jax.Array, jax.Array]:
    R, R_dot = utils._get_R_and_dot(
        phi=state.roll,
        theta=state.pitch,
        psi=state.yaw,
        phi_dot=state.roll_dot,
        theta_dot=state.pitch_dot,
        psi_dot=state.yaw_dot,
    )
    t = jnp.array([state.x, state.y, state.z])
    t_dot = jnp.array([state.x_dot, state.y_dot, state.z_dot])
    return utils._leg_pos_vel(R, t, R_dot, t_dot)


def joint_angles(state: State) -> jax.Array:
    return jnp.concatenate(
        utils._angle_joint(
            x=state.x,
            y=state.y,
            z=state.z,
            phi=state.roll,
            theta=state.pitch,
            psi=state.yaw,
        )
    )


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


def head_xyz_acc_cost_single(
    ref: jax.Array, state: State, control: Control, w: jax.Array
) -> jax.Array:
    """Cost for a single input pairing."""
    R_T = _R_T(state)
    acc = jnp.array([control.x, control.y, control.z])
    world = acc + const.gravity
    head = R_T @ world
    diff = (head - ref) * w
    delta_xy = diff.at[:2].get()
    delta_z = diff.at[2].get()
    return _hyper(delta_xy @ delta_xy) + _hyper(delta_z * delta_z)


def head_xyz_acc_cost_arr(
    weights: Weights,
    acc_ref: jax.Array,
    state: State,
    control: Control,
) -> jax.Array:
    """Head acceleration cost terms."""
    assert acc_ref.ndim == 2
    assert acc_ref.shape[1] == 3
    assert acc_ref.shape[0] == control.x.size

    # skip initial conditions in state
    w = weights.scale_acc(control.size)
    assert w.shape == acc_ref.shape
    single = jax.vmap(head_xyz_acc_cost_single)
    cost_arr = jnp.ravel(single(acc_ref, state.pop0(), control, w))
    return cost_arr


def _head_xyz_acc_cost(
    weights: Weights,
    acc_ref: jax.Array,
    state: State,
    control: Control,
) -> jax.Array:
    """Head acceleration cost."""
    cost_arr = head_xyz_acc_cost_arr(weights, acc_ref, state, control)
    return 0.5 * jnp.sum(jnp.mean(cost_arr, axis=0))


def omega_cost_single(ref: jax.Array, state: State, w: jax.Array) -> jax.Array:
    """Cost for a single input pairing."""
    R_T = _R_T(state)
    R_dot = _R_dot(state)
    omega_mat = R_T @ R_dot
    omega = jnp.array([omega_mat[2, 1], omega_mat[0, 2], omega_mat[1, 0]])
    diff = (omega - ref) * w
    return _hyper(diff @ diff)


def omega_cost_arr(
    weights: Weights,
    omega_ref: jax.Array,
    state: State,
    control: Control,
) -> jax.Array:
    """Angular velocity cost."""
    assert omega_ref.ndim == 2
    assert omega_ref.shape[1] == 3
    assert omega_ref.shape[0] == control.x.size

    # skip initial conditions in state
    w = weights.scale_omega(control.size)
    assert w.shape == omega_ref.shape
    single = jax.vmap(omega_cost_single)
    cost_arr = jnp.ravel(single(omega_ref, state.pop0(), w))
    return cost_arr


def _omega_cost(
    weights: Weights,
    omega_ref: jax.Array,
    state: State,
    control: Control,
) -> jax.Array:
    """Angular velocity cost."""
    cost_arr = omega_cost_arr(weights, omega_ref, state, control)
    return 0.5 * jnp.sum(jnp.mean(cost_arr, axis=0))


def leg_boundary_cost_arr(
    weights: Weights,
    length_cost: quartic_cost.QuarticCost,
    vel_cost: quartic_cost.QuarticCost,
    state: State,
    control: Control,
) -> tuple[jax.Array, jax.Array]:
    """Include leg length and leg velocity costs.

    By using automatic differentiation, we can compute the lengths and the
    velocities cheaper than computing them separately, which is productive.
    """
    lengths, vels = jax.vmap(length_and_vel)(state.pop0())
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
    state: State,
    control: Control,
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


def joint_angle_boundary_cost_arr(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: State,
    control: Control,
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
    state: State,
    control: Control,
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
    state: State,
    control: Control,
) -> jax.Array:
    yaw = state.yaw[1:]
    costs = jax.vmap(cost)(yaw)
    w = weights.scale_yaw(control.size)
    return costs * w


def _yaw_boundary_cost(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state: State,
    control: Control,
) -> jax.Array:
    cost_arr = yaw_boundary_cost_arr(weights, cost, state, control)
    return jnp.sum(jnp.mean(cost_arr, axis=0))


def control_cost_arr(
    weights: Weights,
    control: Control,
) -> jax.Array:
    w = weights.scale_control(control.size)
    costs = jnp.square(control.flatten() * w)
    return costs.reshape(-1, 6)


def _control_cost(
    weights: Weights,
    control: Control,
) -> jax.Array:
    cost_arr = control_cost_arr(weights, control)
    return 0.5 * jnp.sum(jnp.mean(cost_arr, axis=0))


def _cost(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    state0: jax.Array,
    control: Control,
    state: tp.Optional[State] = None,
) -> jax.Array:
    if state is None:
        state = get_state(control, state0)
    cost = jnp.array(0.0)
    cost += _head_xyz_acc_cost(weights, acc_ref, state, control)
    cost += _omega_cost(weights, omega_ref, state, control)
    cost += _leg_boundary_cost(
        weights, cost_terms.leg_cost, cost_terms.leg_vel_cost, state, control
    )
    cost += _joint_angle_boundary_cost(
        weights, cost_terms.joint_angle_cost, state, control
    )
    cost += _yaw_boundary_cost(weights, cost_terms.yaw_cost, state, control)
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
    state0: jax.Array,
    control_flat: jax.Array,
) -> jax.Array:
    control = Control.from_flat(control_flat)
    return _cost(
        weights,
        cost_terms,
        acc_ref,
        omega_ref,
        state0,
        control,
    )


@jax.jit
def cost_and_grad_flat_jax(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    state0: jax.Array,
    control_flat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    cost = functools.partial(
        cost_flat_jax,
        weights,
        cost_terms,
        acc_ref,
        omega_ref,
        state0,
    )
    cost_and_grad = jax.value_and_grad(cost)
    return cost_and_grad(control_flat)


def get_scipy_cost(
    weights: Weights,
    cost_terms: CostTerms,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    state0: jax.Array,
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
    state0 :
        Initial state, as a vector of size 12.

    Returns
    -------
    Cost function, gradient function
    """
    params = (
        weights,
        cost_terms,
        acc_ref,
        omega_ref,
        state0,
    )

    cost_and_grad = functools.partial(cost_and_grad_flat_jax, *params)

    # history
    input_mem: list[np.ndarray] = [np.array(np.nan)]
    cost_mem: list[np.ndarray] = [np.array(np.nan)]
    grad_mem: list[np.ndarray] = [np.array(np.nan)]

    def update_mem(control_flat: np.ndarray) -> None:
        val, grad = cost_and_grad(control_flat)
        input_mem[0] = control_flat
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
