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
import scipy.sparse.linalg as sci_sp_lin

import exp_mpc.stewart_min.const as const
import exp_mpc.stewart_min.utils as utils
import exp_mpc.stewart_min.quartic_cost as quartic_cost

# make sure to enable 64-bit precision for jax
# testing has shown that this yields better performance in constant acceleration
#  test runs
# jax.config.update("jax_enable_x64", True)


###############
# state model #
###############


@jax.jit
def _discrete_1d_nonuniform_euler(
    x0: jax.Array,
    v0: jax.Array,
    a: jax.Array,
    gaps: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Discrete 1D non-uniform Euler integration, with scalar initial data.

    Parameters
    ----------
    x0 :
        Initial position.
    v0 :
        Initial velocity.
    a :
        Constant accelerations.
    gaps :
        Discrete gaps between the control points.
        A gap of 1 means that the control point is at the next time step.

    Returns
    -------
    Integrated position and velocity.

    Notes
    -----
    We perform several Euler steps exactly, with constant acceleration.
    The idea is that the model reduces to consider the perturbed continuous
    double integrator model:

    .. math::

      \begin{bmatrix} \dot{x} \\ \dot{v} \end{bmatrix} =
      \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}
      \begin{bmatrix} x \\ v \end{bmatrix} +
      \begin{bmatrix} \Delta t / 2 \\ 1 \end{bmatrix} a.
    """
    # cf. https://stackoverflow.com/questions/37726830/how-to-determine-if-a-number-is-any-type-of-int-core-or-numpy-signed-or-not
    assert jnp.issubdtype(gaps.dtype, jnp.integer)
    assert jnp.issubdtype(a.dtype, jnp.floating)
    a = jnp.ravel(a)
    gaps = jnp.ravel(gaps)
    assert a.shape == gaps.shape
    v = jnp.cumsum(jnp.concatenate([jnp.array([v0]), const.dt * gaps * a]))
    x_vel = const.dt * gaps * v.at[1:].get()
    x_acc = 0.5 * (gaps - gaps**2) * (const.dt**2) * a
    x = jnp.cumsum(jnp.concatenate([jnp.array([x0]), x_vel + x_acc]))
    return x, v


###############
# dataclasses #
###############


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class State:
    """Parallel arrays for state vector.

    Should be computed from teh Control dataclass.
    """

    x: jax.Array
    y: jax.Array
    z: jax.Array
    roll: jax.Array
    pitch: jax.Array
    yaw: jax.Array
    x_dot: jax.Array
    y_dot: jax.Array
    z_dot: jax.Array
    roll_dot: jax.Array
    pitch_dot: jax.Array
    yaw_dot: jax.Array

    def __iter__(self) -> tp.Iterator[jax.Array]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __getitem__(self, key: int | slice) -> "State":
        return State(*[arr.at[key].get() for arr in self])

    def flatten(self) -> jax.Array:
        """Flatten the state into a single array."""
        return jnp.concatenate([arr.reshape(-1) for arr in self])

    def xyz(self) -> jax.Array:
        """Get the x, y, z coordinates of the state points."""
        # enforce 1d
        x = self.x.reshape(-1)
        y = self.y.reshape(-1)
        z = self.z.reshape(-1)
        return jnp.concatenate([x, y, z])

    def xyz_dot(self) -> jax.Array:
        """Get the x, y, z velocities of the state points."""
        # enforce 1d
        x_dot = self.x_dot.reshape(-1)
        y_dot = self.y_dot.reshape(-1)
        z_dot = self.z_dot.reshape(-1)
        return jnp.concatenate([x_dot, y_dot, z_dot])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Control:
    """Parallel array of control vector."""

    x: jax.Array
    y: jax.Array
    z: jax.Array
    roll: jax.Array
    pitch: jax.Array
    yaw: jax.Array

    @property
    def size(self) -> int:
        """Get the size of the control vector."""
        return self.x.size

    def __iter__(self) -> tp.Iterator[jax.Array]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __getitem__(self, key: int | slice) -> "Control":
        return Control(*[arr.at[key].get() for arr in self])

    def get_state(
        self,
        state0: jax.Array,
        gaps: tp.Optional[jax.Array] = None,
    ) -> State:
        """Convert a control dataclass to a state dataclass."""
        state0 = jnp.ravel(state0)
        assert state0.size == 12
        x0, y0, z0, roll0, pitch0, yaw0 = state0[:6]
        x_dot0, y_dot0, z_dot0, roll_dot0, pitch_dot0, yaw_dot0 = state0[6:]
        if gaps is None:
            gaps = jnp.ones(self.x.size, dtype=int)
        assert gaps.size == self.x.size
        _euler = _discrete_1d_nonuniform_euler
        x, x_dot = _euler(x0, x_dot0, self.x, gaps)
        y, y_dot = _euler(y0, y_dot0, self.y, gaps)
        z, z_dot = _euler(z0, z_dot0, self.z, gaps)
        roll, roll_dot = _euler(roll0, roll_dot0, self.roll, gaps)
        pitch, pitch_dot = _euler(pitch0, pitch_dot0, self.pitch, gaps)
        yaw, yaw_dot = _euler(yaw0, yaw_dot0, self.yaw, gaps)
        return State(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            x_dot=x_dot,
            y_dot=y_dot,
            z_dot=z_dot,
            roll_dot=roll_dot,
            pitch_dot=pitch_dot,
            yaw_dot=yaw_dot,
        )

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
        flat = jnp.reshape(flat, (-1, 6))
        return cls(
            x=flat[:, 0],
            y=flat[:, 1],
            z=flat[:, 2],
            roll=flat[:, 3],
            pitch=flat[:, 4],
            yaw=flat[:, 5],
        )

    def flatten(self) -> jax.Array:
        """Convert a control dataclass to a flat array."""
        return jnp.ravel(
            jnp.column_stack(
                [
                    self.x,
                    self.y,
                    self.z,
                    self.roll,
                    self.pitch,
                    self.yaw,
                ]
            )
        )

    def xyz(self) -> jax.Array:
        """Get the x, y, z coordinates of the control points."""
        # enforce 1d
        x = self.x.reshape(-1)
        y = self.y.reshape(-1)
        z = self.z.reshape(-1)
        return jnp.concatenate([x, y, z])


###################
# helper wrappers #
###################


def _get_R(state: State) -> jax.Array:
    """Get the rotation matrix from the state."""
    return utils._get_R(
        phi=state.roll,
        theta=state.pitch,
        psi=state.yaw,
    )


def _get_R_T(state: State) -> jax.Array:
    """Get the rotation matrix from the state."""
    return jnp.transpose(_get_R(state))


def _get_R_dot(state: State) -> jax.Array:
    """Rotation matrix derivative from the state."""
    return utils._get_R_dot(
        phi=state.roll,
        theta=state.pitch,
        psi=state.yaw,
        phi_dot=state.roll_dot,
        theta_dot=state.pitch_dot,
        psi_dot=state.yaw_dot,
    )


def _get_R_dot2(state: State, control: Control) -> jax.Array:
    """Get the rotation matrix 2nd derivative from the state."""
    return utils._get_R_dot2(
        phi=state.roll,
        theta=state.pitch,
        psi=state.yaw,
        phi_dot=state.roll_dot,
        theta_dot=state.pitch_dot,
        psi_dot=state.yaw_dot,
        phi_dot2=control.roll,
        theta_dot2=control.pitch,
        psi_dot2=control.yaw,
    )


def _get_squared_lengths(state: State) -> jax.Array:
    R = _get_R(state)
    t = state.xyz()

    lengths = []
    for i in range(6):
        top_i = R @ const.tops[i] + t
        diff = top_i - const.bots[i]
        lengths.append(diff @ diff)

    return jnp.array(lengths)


def _get_length_and_vel(state: State) -> tuple[jax.Array, jax.Array]:
    R, R_dot = utils._get_R_and_dot(
        phi=state.roll,
        theta=state.pitch,
        psi=state.yaw,
        phi_dot=state.roll_dot,
        theta_dot=state.pitch_dot,
        psi_dot=state.yaw_dot,
    )
    t = state.xyz()
    t_dot = state.xyz_dot()
    return utils._leg_pos_vel(R, t, R_dot, t_dot)


def _get_joint_angles(state: State) -> jax.Array:
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


################
# weight class #
################


def _init_field(arr: jax.Array) -> jax.Array:
    """Initialize a field with the same shape as the input array."""
    return dataclasses.field(
        default_factory=lambda: jnp.ones(arr.shape),
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
    alpha_acc: jax.Array = _init_field(jnp.ones(1) * 4.0)
    alpha_omega: jax.Array = _init_field(jnp.ones(1) * 4.0)
    alpha_leg: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_leg_vel: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_joint_angle: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_yaw: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_control: jax.Array = _init_field(jnp.ones(1) * 0.0)

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

    def _time_scale(self, n: int, alpha: jax.Array) -> jax.Array:
        """Get time scale weights for flat array."""
        # identity
        # return jnp.ones(n, dtype=float)

        # exponential decrease
        return jnp.exp(-jnp.arange(n, dtype=float) / n * alpha)

    def scale_acc(self, n: int) -> jax.Array:
        """Get scale weights for flat acceleration array."""
        time_scale = self._time_scale(n, self.alpha_acc)
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.acc.size))
        val_scale = jnp.tile(self.acc, (n, 1))
        return time_scale * val_scale

    def scale_omega(self, n: int) -> jax.Array:
        """Get scale weights for flat angular velocity array."""
        time_scale = self._time_scale(n, self.alpha_omega)
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.omega.size))
        val_scale = jnp.tile(self.omega, (n, 1))
        return time_scale * val_scale

    def scale_leg(self, n: int) -> jax.Array:
        """Get scale weights for flat leg length array."""
        time_scale = self._time_scale(n, self.alpha_leg)
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.leg.size))
        val_scale = jnp.tile(self.leg, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_leg_vel(self, n: int) -> jax.Array:
        """Get scale weights for flat leg length array."""
        time_scale = self._time_scale(n, self.alpha_leg_vel)
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.leg_vel.size))
        val_scale = jnp.tile(self.leg_vel, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_joint_angle(self, n: int) -> jax.Array:
        """Get scale weights for flat joint angle array."""
        time_scale = self._time_scale(n, self.alpha_joint_angle)
        time_scale = jnp.tile(
            time_scale.reshape(-1, 1), (1, self.joint_angle.size)
        )
        val_scale = jnp.tile(self.joint_angle, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_yaw(self, n: int) -> jax.Array:
        """Get scale weights for flat yaw angle array."""
        time_scale = self._time_scale(n, self.alpha_yaw)
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.yaw.size))
        val_scale = jnp.tile(self.yaw, (n, 1))
        return jnp.ravel(time_scale * val_scale)

    def scale_control(self, n: int) -> jax.Array:
        """Get scale weights for flat control array."""
        time_scale = self._time_scale(n, self.alpha_control)
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.control.size))
        val_scale = jnp.tile(self.control, (n, 1))
        return jnp.ravel(time_scale * val_scale)


###################
# hyperbolic cost #
###################


def _hyper(x: jax.Array) -> jax.Array:
    """Hyperbolic cost function.

    When composed with the vector L^2 norm, this function is quadratic, but
    has the same linear asymptotics as the L^1 norm.

    Note
    ----
    We modified the hyperbolic tangent function to work better with auto-diff.
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
    # a = 2**-6
    a = 2**0
    return jnp.sqrt(1.0 + x / a) - 1.0


########
# cost #
########


def _head_xyz_acc_cost(
    weights: Weights,
    acc_ref: jax.Array,
    state0: jax.Array,
    control: Control,
) -> jax.Array:
    """Head acceleration cost."""
    assert acc_ref.ndim == 2
    assert acc_ref.shape[1] == 3
    assert acc_ref.shape[0] == control.x.size

    state = control.get_state(state0)

    def _single(
        _ref: jax.Array, _state: State, _control: Control, _w: jax.Array
    ) -> jax.Array:
        """Cost for a single input pairing."""
        _R_T = _get_R_T(_state)
        _R_dot2 = _get_R_dot2(_state, _control)
        _acc = _control.xyz()
        _world = _R_dot2 @ const.human_displacement + _acc + const.gravity
        _head = _R_T @ _world
        _diff = (_head - _ref) * _w
        _delta_z = _diff.at[2].get()
        _delta_xy = _diff.at[:2].get()
        return _hyper(_delta_xy @ _delta_xy) + _hyper(_delta_z * _delta_z)

    # skip initial conditions in state
    w = weights.scale_acc(control.size)
    assert w.shape == acc_ref.shape
    costs = jnp.ravel(jax.vmap(_single)(acc_ref, state[1:], control, w))
    return jnp.mean(costs)


def _get_omega_cost(
    weights: Weights,
    omega_ref: jax.Array,
    state0: jax.Array,
    control: Control,
) -> jax.Array:
    """Angular velocity cost."""
    assert omega_ref.ndim == 2
    assert omega_ref.shape[1] == 3
    assert omega_ref.shape[0] == control.x.size

    state = control.get_state(state0)

    def _single(_ref: jax.Array, _state: State, _w: jax.Array) -> jax.Array:
        """Cost for a single input pairing."""
        _R_T = _get_R_T(_state)
        _R_dot = _get_R_dot(_state)
        _omega_mat = _R_T @ _R_dot
        _omega = jnp.array(
            [_omega_mat[2, 1], _omega_mat[0, 2], _omega_mat[1, 0]]
        )
        _diff = (_omega - _ref) * _w
        return _hyper(_diff @ _diff)

    # skip initial conditions in state
    w = weights.scale_omega(control.size)
    assert w.shape == omega_ref.shape
    costs = jnp.ravel(jax.vmap(_single)(omega_ref, state[1:], w))
    return jnp.mean(costs)


def _get_leg_boundary(
    weights: Weights,
    length_cost: quartic_cost.QuarticCost,
    vel_cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    control: Control,
) -> jax.Array:
    """Include leg length and leg velocity costs.

    By using automatic differentiation, we can compute the lengths and the
    velocities cheaper than computing them separately, which is productive.
    """
    state = control.get_state(state0)
    lengths, vels = jax.vmap(_get_length_and_vel)(state[1:])
    lengths = jnp.ravel(lengths)
    vels = jnp.ravel(vels)
    length_costs = jax.vmap(length_cost)(lengths)
    vel_costs = jax.vmap(vel_cost)(vels)
    w_len = weights.scale_leg(control.size)
    w_vel = weights.scale_leg_vel(control.size)
    return jnp.mean(length_costs * w_len) + jnp.mean(vel_costs * w_vel)


def _get_joint_angle_boundary(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    control: Control,
) -> jax.Array:
    """Joint angle cost.

    This is about 3 times more expensive to compute than the other
    cost functions (including boundary cost functions).
    """
    state = control.get_state(state0)
    angles = jnp.ravel(jax.vmap(_get_joint_angles)(state[1:]))
    costs = jax.vmap(cost)(angles)
    w = weights.scale_joint_angle(control.size)
    return jnp.mean(costs * w)


def _get_yaw_boundary(
    weights: Weights,
    cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    control: Control,
) -> jax.Array:
    state = control.get_state(state0)
    yaw = state.yaw[1:]
    costs = jax.vmap(cost)(yaw)
    w = weights.scale_yaw(control.size)
    return jnp.mean(costs * w)


def _control_cost(
    weights: Weights,
    control: Control,
) -> jax.Array:
    # control_arr = control.flatten().reshape(-1, 6)
    # control_diff = jnp.ravel(jnp.diff(control_arr, axis=0))
    # costs = jnp.square(
    #     jnp.concatenate([control_arr.at[0, :].get(), control_diff])
    # )
    costs = jnp.square(control.flatten())
    w = weights.scale_control(control.size)
    return jnp.mean(costs * w)


def _cost(
    weights: Weights,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    leg_cost: quartic_cost.QuarticCost,
    leg_vel_cost: quartic_cost.QuarticCost,
    joint_angle_cost: quartic_cost.QuarticCost,
    yaw_cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    control: Control,
) -> jax.Array:
    cost = jnp.array(0.0)
    cost = cost + _head_xyz_acc_cost(weights, acc_ref, state0, control)
    cost = cost + _get_omega_cost(weights, omega_ref, state0, control)
    cost = cost + _get_leg_boundary(
        weights, leg_cost, leg_vel_cost, state0, control
    )
    # cost = cost + _get_joint_angle_boundary(
    #     weights, joint_angle_cost, state0, control
    # )
    cost = cost + _get_yaw_boundary(weights, yaw_cost, state0, control)
    cost = cost + _control_cost(weights, control)
    return cost


#######################
# scipy cost wrappers #
#######################

# remark: do not use function closures in jax.jit, because this will trigger
#  recompilation, which takes about 1 second (much longer than the desired
#  5ms run time)


@jax.jit
def cost_flat_jax(
    weights: Weights,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    leg_cost: quartic_cost.QuarticCost,
    leg_vel_cost: quartic_cost.QuarticCost,
    joint_angle_cost: quartic_cost.QuarticCost,
    yaw_cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    control_flat: jax.Array,
) -> jax.Array:
    control = Control.from_flat(control_flat)
    return _cost(
        weights,
        acc_ref,
        omega_ref,
        leg_cost,
        leg_vel_cost,
        joint_angle_cost,
        yaw_cost,
        state0,
        control,
    )


@jax.jit
def cost_jac_jax(
    weights: Weights,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    leg_cost: quartic_cost.QuarticCost,
    leg_vel_cost: quartic_cost.QuarticCost,
    joint_angle_cost: quartic_cost.QuarticCost,
    yaw_cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    control_flat: jax.Array,
) -> jax.Array:
    cost = functools.partial(
        cost_flat_jax,
        weights,
        acc_ref,
        omega_ref,
        leg_cost,
        leg_vel_cost,
        joint_angle_cost,
        yaw_cost,
        state0,
    )
    res = jax.grad(cost)(control_flat)
    return res


@jax.jit
def cost_and_grad_flat_jax(
    weights: Weights,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    leg_cost: quartic_cost.QuarticCost,
    leg_vel_cost: quartic_cost.QuarticCost,
    joint_angle_cost: quartic_cost.QuarticCost,
    yaw_cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    control_flat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    cost = functools.partial(
        cost_flat_jax,
        weights,
        acc_ref,
        omega_ref,
        leg_cost,
        leg_vel_cost,
        joint_angle_cost,
        yaw_cost,
        state0,
    )
    cost_and_grad = jax.value_and_grad(cost)
    return cost_and_grad(control_flat)


@jax.jit
def cost_hessp_jax(
    weights: Weights,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    leg_cost: quartic_cost.QuarticCost,
    leg_vel_cost: quartic_cost.QuarticCost,
    joint_angle_cost: quartic_cost.QuarticCost,
    yaw_cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    control_flat: jax.Array,
    v: jax.Array,
) -> jax.Array:
    cost = functools.partial(
        cost_flat_jax,
        weights,
        acc_ref,
        omega_ref,
        leg_cost,
        leg_vel_cost,
        joint_angle_cost,
        yaw_cost,
        state0,
    )
    # primals and tangents should be _tuples_ of arrays
    primals = (jnp.array(control_flat),)
    tangents = (jnp.array(v),)
    res = jax.jvp(jax.grad(cost), primals, tangents)
    return res[1]


@jax.jit
def cost_hess_mat_jax(
    weights: Weights,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    leg_cost: quartic_cost.QuarticCost,
    leg_vel_cost: quartic_cost.QuarticCost,
    joint_angle_cost: quartic_cost.QuarticCost,
    yaw_cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    control_flat: jax.Array,
) -> jax.Array:
    cost = functools.partial(
        cost_flat_jax,
        weights,
        acc_ref,
        omega_ref,
        leg_cost,
        leg_vel_cost,
        joint_angle_cost,
        yaw_cost,
        state0,
    )
    return jax.hessian(cost)(control_flat)


def get_cost(
    weights: Weights,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    leg_cost: quartic_cost.QuarticCost,
    leg_vel_cost: quartic_cost.QuarticCost,
    joint_angle_cost: quartic_cost.QuarticCost,
    yaw_cost: quartic_cost.QuarticCost,
    state0: jax.Array,
    hess_type: str = "hessp",  # hessp, hess_lin_op, hess_mat
) -> tuple[tp.Callable, tp.Callable, tp.Callable]:
    """Get scipy cost functions.

    Parameters
    ----------
    acc_ref :
        Reference head acceleration.
        Should have shape (n, 3) with n the number of control points.
    omega_ref :
        Reference head angular velocity.
        Should have shape (n, 3) with n the number of control points.
    leg_cost :
        Cost function for leg lengths.
    leg_vel_cost :
        Cost function for leg velocities.
    joint_angle_cost :
        Cost function for joint angles.
    yaw_cost :
        Cost function for yaw angles.
    state0 :
        Initial state, as a vector of size 12.
    hess_type :
        Type of hessian to use.
        Options inlude:
        hessp: hessian-vector product.
        hess_lin_op: linear operator for the hessian.
        hess_mat: full hessian matrix.

    Returns
    -------
    Cost function, gradient function, and hessian function.
    """
    params = (
        weights,
        acc_ref,
        omega_ref,
        leg_cost,
        leg_vel_cost,
        joint_angle_cost,
        yaw_cost,
        state0,
    )

    # warning: deprecated
    # cost = functools.partial(cost_flat_jax, *params)
    # cost_jac = functools.partial(cost_jac_jax, *params)
    # def cost_wrapper(control_flat: np.ndarray) -> float:
    #     return float(cost(control_flat))
    # def cost_jac_wrapper(control_flat: np.ndarray) -> np.ndarray:
    #     return np.array(cost_jac(control_flat))

    cost_and_grad = functools.partial(cost_and_grad_flat_jax, *params)
    cost_hessp = functools.partial(cost_hessp_jax, *params)
    cost_hess_mat = functools.partial(cost_hess_mat_jax, *params)

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
        # if not np.allclose(control_flat, input_mem[0]):
        update_mem(control_flat)
        return float(cost_mem[0])

    def cost_jac_wrapper(control_flat: np.ndarray) -> np.ndarray:
        # warning: not safe for general use
        # if not np.allclose(control_flat, input_mem[0]):
        #     update_mem(control_flat)
        return grad_mem[0]

    def cost_hessp_wrapper(
        control_flat: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        return np.array(cost_hessp(jnp.array(control_flat), jnp.array(v)))

    def cost_hess_lin_op_wrapper(
        control_flat: np.ndarray,
    ) -> sci_sp_lin.LinearOperator:
        matvec = functools.partial(cost_hessp_wrapper, control_flat)
        return sci_sp_lin.LinearOperator(
            shape=(control_flat.size, control_flat.size),
            matvec=matvec,  # type: ignore
            dtype=np.float64,
        )

    def cost_hess_mat_wrapper(control_flat: np.ndarray) -> np.ndarray:
        control = jnp.array(control_flat)
        return np.array(cost_hess_mat(control))

    if hess_type == "hessp":
        cost_hess_wrapper = cost_hessp_wrapper
    elif hess_type == "hess_lin_op":
        cost_hess_wrapper = cost_hess_lin_op_wrapper
    elif hess_type == "hess_mat":
        cost_hess_wrapper = cost_hess_mat_wrapper
    else:
        raise ValueError(f"Unknown hess_type: {hess_type}")

    return cost_wrapper, cost_jac_wrapper, cost_hess_wrapper


if __name__ == "__main__":
    # example usage
    import time
    import scipy.optimize as sci_opt
    import tqdm  # type: ignore

    T = 4.0  # s
    num_steps = int(T / const.dt)
    n = 40

    state0 = jnp.zeros(12, dtype=float)

    weights = Weights()

    acc_ref = jnp.array([1.0, 0.0, 0.0]) + const.gravity  # m/s^2
    acc_ref = jnp.tile(A=acc_ref, reps=(n, 1))
    omega_ref = jnp.array([0.0, 0.0, 0.0])  # rad/s
    omega_ref = jnp.tile(A=omega_ref, reps=(n, 1))

    margins = [0.2, 0.1]
    sizes = [1.0, 2**3, 2**8]
    leg_cost = quartic_cost.QuarticCost.from_bounds(
        margins=margins,
        sizes=sizes,
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

    acados_get_cost = functools.partial(
        get_cost,
        weights=weights,
        acc_ref=acc_ref,
        omega_ref=omega_ref,
        leg_cost=leg_cost,
        leg_vel_cost=leg_vel_cost,
        joint_angle_cost=joint_angle_cost,
        yaw_cost=yaw_cost,
        # state0=state0,
    )

    def train_step(
        state0: jax.Array, last_control: jax.Array
    ) -> tuple[
        jax.Array, jax.Array, utils.TableSol, sci_opt.OptimizeResult, float
    ]:
        """Return next state, computed control (flat), and table solution."""
        cost, cost_jac, _ = acados_get_cost(state0=state0)
        t0 = time.time()
        res = sci_opt.minimize(
            fun=cost,
            x0=np.array(last_control),
            method="L-BFGS-B",
            jac=cost_jac,
            options={
                "maxiter": 16,
                "maxls": 8,
            },
        )
        t1 = time.time()
        control = Control.from_flat(jnp.array(res.x))
        state = control.get_state(state0)
        table_sol = utils.TableSol(
            x=jnp.column_stack(list(state)),
            u=jnp.column_stack(list(control)),
            stats=utils.TableStats(
                time=jnp.array(res.nit),
                status=jnp.array(res.status),
                cost=jnp.array(res.fun),
            ),
        )
        t_tot = t1 - t0
        return state[1].flatten(), control.flatten(), table_sol, res, t_tot

    state0 = jnp.zeros(12, dtype=float)
    last_control = jnp.zeros(6 * n, dtype=float)
    times = []
    sol_list = []
    res_list = []

    for _ in tqdm.tqdm(range(num_steps)):
        state0, last_control, table_sol, res, t_tot = train_step(state0, last_control)
        sol_list.append(table_sol)
        res_list.append(res)
        times.append(t_tot)
