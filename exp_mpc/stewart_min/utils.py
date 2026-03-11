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
import numpy as np

import exp_mpc.stewart_min.robo as robo
import exp_mpc.stewart_min.comp as comp
import exp_mpc.stewart_min.vest as vest


###########
# helpers #
###########


def refinement_m(n: int, dt: float, dtp: float) -> int:
    """See the docstring for `control_refinement`."""
    return int(np.floor(dt * n / dtp))


@functools.partial(jax.vmap, in_axes=[None, None, None, 1])
@functools.partial(jax.vmap, in_axes=[0, None, None, None])
def _control_refinement(
    k: jax.Array, dt: jax.Array, dtp: jax.Array, u: jax.Array
) -> jax.Array:
    ell = jnp.ceil(k * dtp / dt).astype(int)
    s1 = jnp.min(jnp.array([dtp, ell * dt - k * dtp]))
    s2 = jnp.max(jnp.array([0, (k + 1) * dtp - ell * dt]))
    return (s1 * u[ell - 1] + s2 * u[ell]) / dtp


@functools.partial(jax.jit, static_argnames=["dt", "dtp"])
def control_refinement(
    dt: jax.Array, dtp: jax.Array, u: jax.Array
) -> jax.Array:
    r"""Average the controls `u` to get a refinement.

    Usually `dt == dt_mpc` and `dtp == dt`.

    Parameters
    ----------
    dt :
        Uniform time steps corresponding to the controls `u`.
    dtp :
        Refined time steps.
        The implementation only makes sense for `dt >= dtp`.
    u :
        Controls to refine.
        Shape is 2d, with axis==0 corresponding to time.

    Notes
    -----
    Suppose that we have a sequence of controls
    :math:`u_0, u_1, \ldots, u_{n - 1}`, where :math:`u_k` is supposed to be
    applied on :math:`[t_k, t_{k + 1}]`, where
    :math:`t_k = t_0 + k \, \Delta t`.
    Define the (continuous-time, but not continuous) indicator function
    :math:`u(t) = u_k` when :math:`t \in [t_k, t_{k + 1}]`.
    Now suppose that a corresponding sequence
    :math:`\tilde{u}_0, \tilde{u}_1, \ldots, \tilde{u}_{m - 1}` at times
    :math:`\tilde{t}_k = t_0 + k \, \widetilde{\Delta t}` such that

    .. math::

    \tilde{u}_k = \frac{1}{\widetilde{\Delta t}}
    \int_{\tilde{t}_k}^{\tilde{t}_{k + 1}} u(t) \operatorname*{d}\!t.

    Here, :math:`m := \lfloor t_n / \widetilde{\Delta t} \rfloor`.
    We assume that :math:`\widetilde{\Delta t} < \Delta t`, because that
    is what we use in practice and it simplifies the logic somewhat.
    For the problem statement `dt` == :math:`\Delta t` and
    `dtp` == :math:`\widehat{\Delta t}`.

    The outer vmap works with each control axis, and the inner vmap refines each
    of these control axes.
    Note that given the assumption that `dtp < dt`, there can be at most two
    overlapping intervals, which to code makes judicious use of.
    """
    # for correct shapes, we need to tranpose the outputs
    assert len(u.shape) == 2
    n = u.shape[0]
    m = refinement_m(n, dt, dtp)
    k = jnp.arange(m)
    return jnp.transpose(_control_refinement(k, dt, dtp, u))


################
# book-keeping #
################


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class VState:
    """Helper for indexing Vestibular state arrays.

    Internal states for the vestibular model are given by `x_state`, and the
    observed states are `y_state`.
    Indexing conventions are given in the `property` decorators.
    Usually, these arrays are 2D, with the rows representing time.
    """

    x_state: tp.Optional[jax.Array]
    y_state: jax.Array

    @property
    def y_accx(self) -> jax.Array:
        return self.y_state[..., 0]

    @property
    def y_accy(self) -> jax.Array:
        return self.y_state[..., 1]

    @property
    def y_accz(self) -> jax.Array:
        return self.y_state[..., 2]

    @property
    def y_omegax(self) -> jax.Array:
        return self.y_state[..., 3]

    @property
    def y_omegay(self) -> jax.Array:
        return self.y_state[..., 4]

    @property
    def y_omegaz(self) -> jax.Array:
        return self.y_state[..., 5]

    def pop0(self) -> "VState":
        """Create a new VState without the first time step.

        This is useful when the initial state should be ignored in a
        computation.
        """
        assert len(self.y_state.shape) == 2
        if self.x_state is not None:
            assert len(self.x_state.shape) == 2
            assert self.x_state.shape[0] >= 2
            assert self.y_state.shape[0] == self.x_state.shape[0]
            return VState(self.x_state[1:], self.y_state[1:])
        else:
            assert self.y_state.shape[0] >= 2
            return VState(None, self.y_state[1:])

    def get0(self) -> "VState":
        """Usually create a new State with *only* the initial state."""
        if len(self.y_state.shape) == 2:
            x_state = None if self.x_state is None else self.x_state[0]
            return VState(x_state, self.y_state[0])
        else:
            return self

    @property
    def size(self) -> int:
        """Get number of time steps that control represents."""
        if len(self.y_state.shape) == 2:
            return self.y_state.shape[0]
        else:
            return 1  # 1 time step


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RState:
    """Helper for indexing a Robot state arrays.

    Usually, the `state` attribute should be 2D, with rows representing time
    and columns representing state values.
    Sometimes, `state` may represent a single time point, in which case the
    array is 1D.
    These properties are enforced in `__post_init__`.
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

    def pop0(self) -> "RState":
        """Create a new RState without the first time step.

        This is useful when the initial state should be ignored in a
        computation.
        """
        assert len(self.state.shape) == 2
        assert self.state.shape[0] >= 2
        return RState(self.state[1:])

    def get0(self) -> "RState":
        """Usually create a new State with *only* the initial state."""
        if len(self.state.shape) == 2:
            return RState(self.state[0])
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

    def refine_control(
        self, dt: jax.Array | float, dtp: jax.Array | float
    ) -> "Control":
        """Get averaged refined controls.

        Parameters
        ----------
        dt :
            Time step corresponding to current controls.
        dtp :
            Time step for refined controls.
            Assume that `dt >= dtp`.

        Returns
        -------
        Refined controls.
        """
        assert len(self.control.shape) == 2
        return Control(control_refinement(dt, dtp, self.control))


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

    x: RState
    u: Control
    vstate_irl: VState
    vstate_sim: VState
    stats: TableStats


############
# wrappers #
############


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot(state: RState, use_xy: bool = True) -> jax.Array:
    """Get the rotation matrix."""
    assert state.size == 1
    return comp.rot(state.roll, state.pitch, state.yaw, use_xy)


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot_dot(state: RState, use_xy: bool = True) -> jax.Array:
    """Get the rotation matrix derivative."""
    assert state.size == 1
    return comp.rot_dot(
        state.roll,
        state.pitch,
        state.yaw,
        state.roll_dot,
        state.pitch_dot,
        state.yaw_dot,
        use_xy,
    )


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot_and_dot(state: RState, use_xy: bool = True) -> jax.Array:
    """Get the rotation matrix derivative."""
    assert state.size == 1
    return comp.rot_and_dot(
        state.roll,
        state.pitch,
        state.yaw,
        state.roll_dot,
        state.pitch_dot,
        state.yaw_dot,
        use_xy,
    )


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot_dot2(state: RState, control: Control, use_xy: bool = True) -> jax.Array:
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
        use_xy,
    )


@functools.partial(jax.jit, static_argnames=["robo_geom", "use_rotary"])
def leg_pos(
    state: RState,
    robo_geom: robo.RoboGeom,
    use_rotary: bool = True,
) -> jax.Array:
    """All leg lengths."""
    assert state.size == 1
    R = rot(state, use_rotary)
    t = jnp.array([state.x, state.y, state.z])
    return comp.leg_pos(robo_geom, R, t)


@functools.partial(jax.jit, static_argnames=["robo_geom", "use_rotary"])
def leg_vel(
    state: RState,
    robo_geom: robo.RoboGeom,
    use_rotary: bool = True,
) -> jax.Array:
    """All leg velocities."""
    assert state.size == 1
    R, R_dot = rot_and_dot(state, use_rotary)
    t = jnp.array([state.x, state.y, state.z])
    t_dot = jnp.array([state.x_dot, state.y_dot, state.z_dot])
    return comp.leg_vel(robo_geom, R, t, R_dot, t_dot)


@functools.partial(jax.jit, static_argnames=["robo_geom", "use_rotary"])
def leg_pos_vel(
    state: RState,
    robo_geom: robo.RoboGeom,
    use_rotary: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """All leg lengths and velocities."""
    assert state.size == 1
    R, R_dot = rot_and_dot(state, use_rotary)
    t = jnp.array([state.x, state.y, state.z])
    t_dot = jnp.array([state.x_dot, state.y_dot, state.z_dot])
    return comp.leg_pos_vel(robo_geom, R, t, R_dot, t_dot)


@functools.partial(jax.jit, static_argnames=["robo_geom", "use_rotary"])
def leg_acc(
    state: RState,
    control: Control,
    robo_geom: robo.RoboGeom,
    use_rotary: bool = True,
) -> jax.Array:
    """All leg accelerations."""
    assert state.size == 1
    assert control.size == 1
    R, R_dot = rot_and_dot(state, use_rotary)
    R_dot2 = rot_dot2(state, control, use_rotary)
    t = jnp.array([state.x, state.y, state.z])
    t_dot = jnp.array([state.x_dot, state.y_dot, state.z_dot])
    t_dot2 = jnp.array([control.x, control.y, control.z])
    return comp.leg_acc(robo_geom, R, t, R_dot, t_dot, R_dot2, t_dot2)


@functools.partial(jax.jit, static_argnames=("world",))
def transfer_PHI(state: RState, world: bool = False) -> jax.Array:
    """Matrix to map table euler angle derivatives to head angular velocity."""
    assert state.size == 1
    return comp.transfer_PHI(state.roll, state.pitch, state.yaw, world)


@functools.partial(jax.jit, static_argnames=("world",))
def angle_vel(state: RState, world: bool = False) -> jax.Array:
    """Angular velocity."""
    assert state.size == 1
    angles = [state.roll, state.pitch, state.yaw]
    angles_dot = [state.roll_dot, state.pitch_dot, state.yaw_dot]
    inputs = angles + angles_dot + [world]
    return comp.angle_vel(*inputs)


@functools.partial(jax.jit, static_argnames=("world",))
def angle_acc(
    state: RState, control: Control, world: bool = False
) -> jax.Array:
    """Angular acceleration."""
    assert state.size == 1
    assert control.size == 1
    angles = [state.roll, state.pitch, state.yaw]
    angles_dot = [state.roll_dot, state.pitch_dot, state.yaw_dot]
    angles_dot2 = [control.roll, control.pitch, control.yaw]
    inputs = angles + angles_dot + angles_dot2 + [world]
    return comp.angle_acc(*inputs)


@functools.partial(jax.jit, static_argnames=["robo_geom", "use_xy"])
def angle_joint(
    state: RState,
    robo_geom: robo.RoboGeom = robo.RoboGeom(),
    use_xy: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Angles at joints."""
    assert state.size == 1
    s = state
    return comp.angle_joint(
        robo_geom,
        s.x,
        s.y,
        s.z,
        s.roll,
        s.pitch,
        s.yaw,
        use_xy,
    )


@functools.partial(jax.jit, static_argnames=["robo_geom", "use_xy"])
def angle_joint_top(
    state: RState,
    robo_geom: robo.RoboGeom = robo.RoboGeom(),
    use_xy: bool = True,
) -> jax.Array:
    """Angles at top joints."""
    assert state.size == 1
    joint_top, _ = angle_joint(state, robo_geom, use_xy)
    return joint_top


@functools.partial(jax.jit, static_argnames=["robo_geom", "use_xy"])
def angle_joint_bot(
    state: RState,
    robo_geom: robo.RoboGeom = robo.RoboGeom(),
    use_xy: bool = True,
) -> jax.Array:
    """Agnles at bottom joints."""
    assert state.size == 1
    _, joint_bot = angle_joint(state, robo_geom, use_xy)
    return joint_bot


###############
# conversions #
###############


def get_rstate(
    dt: jax.Array,
    control: Control,
    rstate0: jax.Array,
) -> RState:
    """Get robot state from controls.

    Parameters
    ----------
    control :
        Robot controls.
    rstate0 :
        Current robot state.
        (Not the initial state from the previous iteration.
        Namely, dissimilar to the initial states in `get_vstate`, below.)

    Returns
    --
    """
    rstate0 = jnp.ravel(rstate0)
    assert rstate0.size == 12
    x0, y0, z0, roll0, pitch0, yaw0 = rstate0[:6]
    x_dot0, y_dot0, z_dot0, roll_dot0, pitch_dot0, yaw_dot0 = rstate0[6:]

    x, x_dot = comp.discrete_1d_euler(dt, x0, x_dot0, control.x)
    y, y_dot = comp.discrete_1d_euler(dt, y0, y_dot0, control.y)
    z, z_dot = comp.discrete_1d_euler(dt, z0, z_dot0, control.z)
    roll, roll_dot = comp.discrete_1d_euler(dt, roll0, roll_dot0, control.roll)
    pitch, pitch_dot = comp.discrete_1d_euler(
        dt, pitch0, pitch_dot0, control.pitch
    )
    yaw, yaw_dot = comp.discrete_1d_euler(dt, yaw0, yaw_dot0, control.yaw)

    non_dots = [x, y, z, roll, pitch, yaw]
    dots = [x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot]
    state = jnp.transpose(jnp.vstack(non_dots + dots))
    return RState(state)


def get_vstate(
    vspec_acc: vest.VSpec,
    vspec_omega: vest.VSpec,
    acc_ctrl: jax.Array,
    omega_ctrl: jax.Array,
    vstate0: jax.Array,
) -> VState:
    """Get vestibular state dataclass.

    Parameters
    ----------
    vspec :
        Implicitly determines which time step we are using.
        E.g., are we using sim or irl.
    acc_ctrl :
        Control linear accelerations, for vestibular model.
        (Acutal linear accelerations, before vestibular processing.)
    omega_ctrl :
        Control angular velocities, for vestibular model.
        (Acutal angular velocities, before vestibular processing.)
    vstate0 :
        Initial vestibular state.

    Returns
    -------
    Vestibular state.
    """
    s_acc = vspec_acc
    s_ome = vspec_omega

    vstate0 = jnp.ravel(vstate0)
    x_num_acc = s_acc.E0.shape[0]
    x_num_omega = s_ome.E1.shape[0]
    assert vstate0.size == 3 * (x_num_acc + x_num_omega)

    acc0 = vstate0[: 3 * x_num_acc]
    accx0, accy0, accz0 = list(acc0.reshape(-1, x_num_acc))
    omega0 = vstate0[3 * x_num_acc :]
    omegax0, omegay0, omegaz0 = list(omega0.reshape(-1, x_num_omega))
    acc_lti_int = functools.partial(
        comp.lti_int, s_acc.E0, s_acc.E1, s_acc.C, s_acc.D
    )
    omega_lti_int = functools.partial(
        comp.lti_int, s_ome.E0, s_ome.E1, s_ome.C, s_ome.D
    )

    x_accx, y_accx = acc_lti_int(accx0, acc_ctrl[:, 0])
    x_accy, y_accy = acc_lti_int(accy0, acc_ctrl[:, 1])
    x_accz, y_accz = acc_lti_int(accz0, acc_ctrl[:, 2])

    x_omegax, y_omegax = omega_lti_int(omegax0, omega_ctrl[:, 0])
    x_omegay, y_omegay = omega_lti_int(omegay0, omega_ctrl[:, 1])
    x_omegaz, y_omegaz = omega_lti_int(omegaz0, omega_ctrl[:, 2])

    x_state = jnp.hstack([x_accx, x_accy, x_accz, x_omegax, x_omegay, x_omegaz])
    y_state = jnp.transpose(
        jnp.vstack([y_accx, y_accy, y_accz, y_omegax, y_omegay, y_omegaz])
    )
    return VState(x_state, y_state)


def _head_acc(rstate: RState, acc: jax.Array) -> jax.Array:
    assert rstate.size == 1  # one time step
    assert len(acc.shape) == 1
    assert acc.size == 3
    R = rot(rstate, use_xy=False)
    return R.T @ (acc + robo.gravity)


def get_vstate_irl(
    vspec_acc: vest.VSpec,
    vspec_omega: vest.VSpec,
    rstate: RState,
    control: Control,
    control0: jax.Array,
    vstate0: jax.Array,
) -> VState:
    """Get vestibular state dataclass, from robot information.

    Parameters
    ----------
    rstate :
        Robot states.
    control :
        Robot controls.
    control0 :
        Control applied during previous iteration.
    vstate0 :
        Vestibular state from previous iteration.

    Returns
    -------
    Vestibular state.
    """
    assert len(control0.shape) == 1
    assert control0.size == 6

    # hack: add initial control to current control, for conventional purposes
    #  see below convention
    # also, note that we need to add gravity, because the robot control lacks
    #  this information
    lin_accs = [control0[:3].reshape(1, -1), control.control[:, :3]]
    acc_ctrl = jnp.vstack(lin_accs)
    acc_ctrl = jax.vmap(_head_acc)(rstate, acc_ctrl)
    omega_ctrl = jax.vmap(angle_vel)(rstate)
    vstate = get_vstate(vspec_acc, vspec_omega, acc_ctrl, omega_ctrl, vstate0)

    # convention: technically, vstate0 represents the initial state from the
    #  previous mpc calculation, and not the initial state for the current run
    # namely, a control was applied between the two values
    return vstate.pop0()


def get_states_with_eigen(
    dt: jax.Array,
    vspec_acc: vest.VSpec,
    vspec_omega: vest.VSpec,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    rstate0: jax.Array,
    vstate0_irl: jax.Array,
    vstate0_sim: jax.Array,
    control0: jax.Array,
    control: Control,
) -> tuple[RState, VState, VState]:
    """Return states, using eigenvector state-space computations.

    WARNING: the returned VState internal states should be interpreted as
    eigen-states.
    To get the correct internal states, you need to transform `P @ x` where the
    columns of `P` are the eigenvectors of `E0`.
    """
    s_acc = vspec_acc
    s_ome = vspec_omega

    # get irl controls
    control_with0 = Control.from_flat(jnp.vstack([control0, control.control]))
    rstate = get_rstate(dt, control, jnp.array(rstate0))
    acc_irl = jax.vmap(_head_acc)(rstate, control_with0.control[:, :3])
    omega_irl = jax.vmap(angle_vel)(rstate)

    # partition
    a_num = 3 * s_acc.E0.shape[0]
    w_num = 3 * s_ome.E0.shape[0]
    v0_irl_a = vstate0_irl[:a_num]
    v0_irl_w = vstate0_irl[a_num:]
    v0_sim_a = vstate0_sim[:a_num]
    v0_sim_w = vstate0_sim[a_num:]

    # initial irl update (for closed feedback)
    v0_irl_a = comp.lti_int_single(s_acc.E0, s_acc.E1, v0_irl_a, acc_irl[0])
    v0_irl_w = comp.lti_int_single(s_ome.E0, s_ome.E1, v0_irl_w, omega_irl[0])
    acc_irl = acc_irl[1:]
    omega_irl = omega_irl[1:]

    # setup general states and controls
    v0_a = jnp.concatenate([v0_irl_a, v0_sim_a])
    v0_w = jnp.concatenate([v0_irl_w, v0_sim_w])
    u_a = jnp.hstack([acc_irl, acc_ref])
    u_w = jnp.hstack([omega_irl, omega_ref])

    # integrate
    x_a, y_a = comp.eigen_int(
        s_acc.eig, s_acc.EP1, s_acc.CP, s_acc.D, s_acc.P_inv, v0_a, u_a
    )
    x_w, y_w = comp.eigen_int(
        s_ome.eig, s_ome.EP1, s_ome.CP, s_ome.D, s_ome.P_inv, v0_w, u_w
    )

    # res
    x_irl = jnp.hstack([x_a[:, :a_num], x_w[:, :w_num]])
    y_irl = jnp.hstack([y_a[:, :3], y_w[:, :3]])
    x_sim = jnp.hstack([x_a[:, a_num:], x_w[:, w_num:]])
    y_sim = jnp.hstack([y_a[:, 3:], y_w[:, 3:]])
    return rstate, VState(x_irl, y_irl), VState(x_sim, y_sim)


################
# viz wrappers #
################


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
    return R.T @ (acc + robo.gravity)


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
