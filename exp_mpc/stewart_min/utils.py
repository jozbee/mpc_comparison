"""Numpy/JAX computations related to the Stewart platform.

Given a table solution, we compute
 * leg pos, leg vel, leg acc
 * angle pos, angle vel, angle acc
 * table acc
 * table joints
 * human vel, human acc.

In general, we implement a (private) jax implementation and a (public) numpy
implementation.
"""
import functools
import dataclasses
import typing as tp

import jax
import jax.numpy as jnp

import exp_mpc.stewart_min.const as const
import exp_mpc.stewart_min.comp as comp


################
# book-keeping #
################


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

    @property
    def size(self) -> int:
        """Get number of time steps that control represents."""
        if len(self.state.shape) == 2:
            return self.state.shape[0]
        else:
            return 1  # 1 time step

    def flatten(self) -> jax.Array:
        return jnp.ravel(self.state)

    def pop0(self) -> "State":
        """Create a new State without the first time step.

        This is useful when the initial state should be ignored in a
        computation.
        """
        assert len(self.state.shape) == 2
        assert self.state.shape[0] >= 2
        return State(self.state[1:])

    def get0(self) -> "State":
        """Usually create a new State with *only* the initial state."""
        if len(self.state.shape) == 2:
            return State(self.state[0])
        else:
            return self


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

    def get0(self) -> "Control":
        """Usually create a new Control with *only* the initial control."""
        if len(self.control.shape) == 2:
            return Control(self.control[0])
        else:
            return self


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TableStats:
    """Statistics of the solution to the Stewart platform OCP."""

    time: jax.Array
    status: jax.Array
    cost: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TableSol:
    """A solution to the Stewart platform OCP."""

    x: State
    u: Control
    stats: TableStats


###############
# conversions #
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


############
# wrappers #
############


@jax.jit
def rot(state: State) -> jax.Array:
    """Get the rotation matrix."""
    assert state.size == 1
    return comp.rot(state.roll, state.pitch, state.yaw)


@jax.jit
def rot_T(state: State) -> jax.Array:
    return jnp.transpose(rot(state))


@jax.jit
def rot_dot(state: State) -> jax.Array:
    """Get the rotation matrix derivative."""
    assert state.size == 1
    return comp.rot_dot(
        state.roll,
        state.pitch,
        state.yaw,
        state.roll_dot,
        state.pitch_dot,
        state.yaw_dot,
    )


@jax.jit
def rot_and_dot(state: State) -> jax.Array:
    """Get the rotation matrix derivative."""
    assert state.size == 1
    return comp.rot_and_dot(
        state.roll,
        state.pitch,
        state.yaw,
        state.roll_dot,
        state.pitch_dot,
        state.yaw_dot,
    )


@jax.jit
def rot_dot2(state: State, control: Control) -> jax.Array:
    """Get the second derivative of the rotation matrix."""
    assert state.size == 1
    assert control.size == 1
    return comp.rot_dot2(
        state.roll,
        state.pitch,
        state.yaw,
        state.roll_dot,
        state.pitch_dot,
        state.yaw_dot,
        control.roll,  # acc
        control.pitch,  # acc
        control.yaw,  # acc
    )


@jax.jit
def leg_pos(state: State) -> jax.Array:
    """All leg lengths."""
    assert state.size == 1
    R = rot(state)
    t = jnp.array([state.x, state.y, state.z])
    return comp.leg_pos(R, t)


@jax.jit
def leg_vel(state: State) -> jax.Array:
    """All leg velocities."""
    assert state.size == 1
    R, R_dot = rot_and_dot(state)
    t = jnp.array([state.x, state.y, state.z])
    t_dot = jnp.array([state.x_dot, state.y_dot, state.z_dot])
    return comp.leg_vel(R, t, R_dot, t_dot)


@jax.jit
def leg_pos_vel(state: State) -> tuple[jax.Array, jax.Array]:
    """All leg lengths and velocities."""
    assert state.size == 1
    R, R_dot = rot_and_dot(state)
    t = jnp.array([state.x, state.y, state.z])
    t_dot = jnp.array([state.x_dot, state.y_dot, state.z_dot])
    return comp.leg_pos_vel(R, t, R_dot, t_dot)


@jax.jit
def leg_acc(state: State, control: Control) -> jax.Array:
    """All leg accelerations."""
    assert state.size == 1
    assert control.size == 1
    R, R_dot = rot_and_dot(state)
    R_dot2 = rot_dot2(state, control)
    t = jnp.array([state.x, state.y, state.z])
    t_dot = jnp.array([state.x_dot, state.y_dot, state.z_dot])
    t_dot2 = jnp.array([control.x, control.y, control.z])
    return comp.leg_acc(R, t, R_dot, t_dot, R_dot2, t_dot2)


@functools.partial(jax.jit, static_argnames=("world",))
def tranfer_PHI(state: State, world: bool = False) -> jax.Array:
    """Matrix to map table euler angle derivatives to head angular velocity."""
    assert state.size == 1
    return comp.transfer_PHI(state.roll, state.pitch, state.yaw, world)


@functools.partial(jax.jit, static_argnames=("world",))
def angle_vel(state: State, world: bool = False) -> jax.Array:
    """Angular velocity."""
    assert state.size == 1
    angles = [state.roll, state.pitch, state.yaw]
    angles_dot = [state.roll_dot, state.pitch_dot, state.yaw_dot]
    inputs = angles + angles_dot + [world]
    return comp.angle_vel(*inputs)


@functools.partial(jax.jit, static_argnames=("world",))
def angle_acc(state: State, control: Control, world: bool = False) -> jax.Array:
    """Angular acceleration."""
    assert state.size == 1
    assert control.size == 1
    angles = [state.roll, state.pitch, state.yaw]
    angles_dot = [state.roll_dot, state.pitch_dot, state.yaw_dot]
    angles_dot2 = [control.roll, control.pitch, control.yaw]
    inputs = angles + angles_dot + angles_dot2 + [world]
    return comp.angle_acc(*inputs)


@jax.jit
def angle_joint(state: State) -> tuple[jax.Array, jax.Array]:
    """Angles at joints."""
    assert state.size == 1
    s = state
    return comp.angle_joint(s.x, s.y, s.z, s.roll, s.pitch, s.yaw)


@jax.jit
def angle_joint_top(state: State) -> jax.Array:
    """Angles at top joints."""
    assert state.size == 1
    joint_top, _ = angle_joint(state)
    return joint_top


@jax.jit
def angle_joint_bot(state: State) -> jax.Array:
    """Agnles at bottom joints."""
    assert state.size == 1
    _, joint_bot = angle_joint(state)
    return joint_bot


###############
# viz helpers #
###############


@jax.jit
def human_angle_vel(sol: TableSol) -> jax.Array:
    """Human angular velocity."""
    state0 = sol.x.get0()
    return angle_vel(state0)


@jax.jit
def table_angle_vel(sol: TableSol) -> jax.Array:
    """Table angular velocity."""
    state0 = sol.x.get0()
    return angle_vel(state0, world=True)


@jax.jit
def human_angle_acc(sol: TableSol) -> jax.Array:
    """Angular acceleration."""
    state0 = sol.x.get0()
    control0 = sol.u.get0()
    return angle_acc(state0, control0)


@jax.jit
def table_angle_acc(sol: TableSol) -> jax.Array:
    """Table angular acceleration."""
    state0 = sol.x.get0()
    control0 = sol.u.get0()
    return angle_acc(state0, control0, world=True)


@jax.jit
def table_angle(sol: TableSol) -> jax.Array:
    """Table angle."""
    state0 = sol.x.get0()
    rpy = jnp.array([state0.roll, state0.pitch, state0.yaw])
    return jnp.degrees(rpy)


@jax.jit
def table_pos(sol: TableSol) -> jax.Array:
    """Table position."""
    state0 = sol.x.get0()
    return jnp.array([state0.x, state0.y, state0.z])


@jax.jit
def table_vel(sol: TableSol) -> jax.Array:
    """Table velocity."""
    state0 = sol.x.get0()
    return jnp.array([state0.x_dot, state0.y_dot, state0.z_dot])


@jax.jit
def table_acc(sol: TableSol) -> jax.Array:
    """Table acceleration."""
    control0 = sol.u.get0()
    return jnp.array([control0.x, control0.y, control0.z])


@jax.jit
def human_vel(sol: TableSol) -> jax.Array:
    """Human velocity."""
    state0 = sol.x.get0()
    R = rot(state0)
    vel = jnp.array([state0.x_dot, state0.y_dot, state0.z_dot])
    return R.T @ vel


@jax.jit
def human_acc(sol: TableSol) -> jax.Array:
    """Human acceleration."""
    state0 = sol.x.get0()
    control0 = sol.u.get0()
    acc = jnp.array([control0.x, control0.y, control0.z])
    R = rot(state0)
    return R.T @ (acc + const.gravity)


@functools.partial(jax.jit, static_argnames=["fun"])
def sol_vmap(
    fun: tp.Callable[[TableSol], jax.Array], sol: TableSol
) -> jax.Array:
    sol.x = sol.x.pop0()  # skip initial condition
    leaves, treedef = jax.tree_util.tree_flatten(sol)

    def flat_fun(*args) -> jax.Array:
        sol = jax.tree_util.tree_unflatten(treedef, args)
        return fun(sol)

    in_axes = [0, 0] + [None] * (len(leaves) - 2)  # probably 3 `None`s
    return jax.vmap(flat_fun, in_axes)(*leaves)


@jax.jit
def human_vel_horizon(sol: TableSol) -> jax.Array:
    """Human velocity over the MPC horizon."""
    return sol_vmap(human_vel, sol)


@jax.jit
def human_angle_vel_horizon(sol: TableSol) -> jax.Array:
    """Human angular velocity over the MPC horizon."""
    return sol_vmap(human_angle_vel, sol)


@jax.jit
def human_acc_horizon(sol: TableSol) -> jax.Array:
    """Human acceleration over the MPC horizon."""
    return sol_vmap(human_acc, sol)


@jax.jit
def human_angle_acc_horizon(sol: TableSol) -> jax.Array:
    """Human angular acceleration over the MPC horizon."""
    return sol_vmap(human_angle_acc, sol)
