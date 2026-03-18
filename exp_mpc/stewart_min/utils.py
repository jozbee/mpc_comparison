"""
Bookkeeping utilities for the MPC algorithm.
Namely, we provide

* bookkeeping classes,
* conversion routines, and
* wrappers to :mod:`exp_mpc.stewart_min.comp`.

Nothing in ``utils.py`` is novel or strictly necessary.
Everything is designed for convenience and documenation purposes.
E.g., the various dataclasses are designed to document the data layout in large
flat arrays, and then corresponding wrappers to :mod:`exp_mpc.stewart_min.comp`
are defined.
Many of the functions were soley written visualization routines.
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
    """Number of refined controls for the new time step..

    Parameters
    ----------
    n : int
        Number of control steps.
    dt : float
        Uniform time steps corresponding to the controls `u`.
    dtp : float
        Refined time steps.
        We assume that ``dtp < dt``.

    Returns
    -------
    m :
        Number of refined control steps.

    See Also
    --------
    :func:`exp_mpc.stewart_min.utils.control_refinement` :
        Function that actually refines the controls.
    """
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
    r"""Average the controls ``u`` to get a refinement.

    Usually ``dt == dt_mpc`` and ``dtp == dt``.

    Parameters
    ----------
    dt :
        Uniform time steps corresponding to the controls ``u``.
    dtp :
        Refined time steps.
        The implementation only makes sense for ``dt >= dtp``.
    u :
        Controls to refine.
        Shape is 2d, with ``axis == 0`` corresponding to time.

    Returns
    -------
    u_refined :
        Refined controls for new time step ``dtp``.

    Notes
    -----
    Suppose that we have a sequence of controls
    :math:`u_0, u_1, \ldots, u_{n - 1}`, where :math:`u_k` is supposed to be
    applied on :math:`[t_k, t_{k + 1}]`, :math:`t_k = t_0 + k \, \Delta t`.
    Define the (continuous-time, but not continuous) step function
    :math:`u(t) = u_k` when :math:`t \in [t_k, t_{k + 1}]`.
    Now define a corresponding sequence
    :math:`\tilde{u}_0, \tilde{u}_1, \ldots, \tilde{u}_{m - 1}` at times
    :math:`\tilde{t}_k = t_0 + k \, \widetilde{\Delta t}` such that

    .. math::

        \tilde{u}_k = \frac{1}{\widetilde{\Delta t}}
        \int_{\tilde{t}_k}^{\tilde{t}_{k + 1}} u(t) \operatorname*{d}\!t.

    Here, :math:`m := \lfloor t_n / \widetilde{\Delta t} \rfloor`.
    We assume that :math:`\widetilde{\Delta t} < \Delta t`, because that
    is what we use in practice and it simplifies the logic somewhat.
    For the problem statement ``dt`` == :math:`\Delta t` and
    ``dtp`` == :math:`\widehat{\Delta t}`.

    The outer :func:`jax.vmap` works with each control axis, and the inner
    :func:`jax.vmap` refines each of these control axes.
    Note that given the assumption that ``dtp < dt``, there can be at most two
    overlapping intervals.
    The code makes judicious use of this observation.
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

    Indexing conventions are given in the ``property`` decorators.

    Parameters
    ----------
    x_state :
        Internal vestibular states.
        In general, should be shaped as a 2D array with time representing rows.
    y_state :
        Observed vestibular states.
        In general, should be shaped as a 2D array with time representing rows.
    """

    x_state: tp.Optional[jax.Array]
    y_state: jax.Array

    @property
    def y_accx(self) -> jax.Array:
        """Observed variables for linear x-acceleration."""
        return self.y_state[..., 0]

    @property
    def y_accy(self) -> jax.Array:
        """Observed variables for linear y-acceleration."""
        return self.y_state[..., 1]

    @property
    def y_accz(self) -> jax.Array:
        """Observed variables for linear z-acceleration."""
        return self.y_state[..., 2]

    @property
    def y_omegax(self) -> jax.Array:
        """Observed variables for angular x-velocity."""
        return self.y_state[..., 3]

    @property
    def y_omegay(self) -> jax.Array:
        """Observed variables for angular y-velocity."""
        return self.y_state[..., 4]

    @property
    def y_omegaz(self) -> jax.Array:
        """Observed variables for angular z-velocity."""
        return self.y_state[..., 5]

    def pop0(self) -> "VState":
        """Create a new VState without the first time step.

        This is useful when the initial state should be ignored in a
        computation.

        Returns
        -------
        vstate :
            New ``Vstate`` with the first states removed.
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
        """Usually create a new VState with **only** the initial state.

        Returns
        -------
        vstate0 :
            ``VState`` object with only the initial data.
            Namely, ``x_data`` and ``y_data`` are 1D.
        """
        if len(self.y_state.shape) == 2:
            x_state = None if self.x_state is None else self.x_state[0]
            return VState(x_state, self.y_state[0])
        else:
            return self

    @property
    def size(self) -> int:
        """Get number of time steps that control represents.

        Returns
        -------
        n :
            Numbner of time steps, which might be ``1`` if the data only
            includes one time step (when data isn't 2D).
        """
        if len(self.y_state.shape) == 2:
            return self.y_state.shape[0]
        else:
            return 1  # 1 time step


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RState:
    """Helper for indexing a Robot state arrays.

    Usually, the ``state`` attribute should be 2D, with rows representing time
    and columns representing state values.
    Sometimes, ``state`` may represent a single time point, in which case the
    array is 1D.
    These properties are enforced in ``__post_init__``.
    See the various ``property`` decorators for specific indexing conventions.

    Note that the initial state is usually stored, so we have a special method
    to ignore it, cf. ``pop0``.

    Parameters
    ----------
    state :
        State variables.
        See the properties.
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
        """X-coordinates."""
        return self.state[..., 0]

    @property
    def y(self) -> jax.Array:
        """Y-coordinates."""
        return self.state[..., 1]

    @property
    def z(self) -> jax.Array:
        """Z-coordinates."""
        return self.state[..., 2]

    @property
    def roll(self) -> jax.Array:
        """Roll coordinates."""
        return self.state[..., 3]

    @property
    def pitch(self) -> jax.Array:
        """Pitch coordinates."""
        return self.state[..., 4]

    @property
    def yaw(self) -> jax.Array:
        """Yaw coordinates."""
        return self.state[..., 5]

    @property
    def x_dot(self) -> jax.Array:
        """X-velocity."""
        return self.state[..., 6]

    @property
    def y_dot(self) -> jax.Array:
        """Y-velocity."""
        return self.state[..., 7]

    @property
    def z_dot(self) -> jax.Array:
        """Z-velocity."""
        return self.state[..., 8]

    @property
    def roll_dot(self) -> jax.Array:
        """Roll velocity."""
        return self.state[..., 9]

    @property
    def pitch_dot(self) -> jax.Array:
        """Pitch velocity."""
        return self.state[..., 10]

    @property
    def yaw_dot(self) -> jax.Array:
        """Yaw velocity."""
        return self.state[..., 11]

    @property
    def size(self) -> int:
        """Get number of time steps that states represents.

        Returns
        -------
        n :
            Numbner of time steps, which might be ``1`` if the data only
            includes one time step (when data isn't 2D).
        """

        if len(self.state.shape) == 2:
            return self.state.shape[0]
        else:
            return 1  # 1 time step

    def flatten(self) -> jax.Array:
        """Get flattened states.

        Useful if the user only want the data and not the dataclass wrapper.

        Returns
        -------
        flat :
            Robot states as a flat array.
        """
        return jnp.ravel(self.state)

    def pop0(self) -> "RState":
        """Create a new RState without the first time step.

        This is useful when the initial state should be ignored in a
        computation.

        Returns
        -------
        rstate :
            New ``Rstate`` with the first states removed.
        """
        assert len(self.state.shape) == 2
        assert self.state.shape[0] >= 2
        return RState(self.state[1:])

    def get0(self) -> "RState":
        """Usually create a new RState with **only** the initial state.

        Returns
        -------
        rstate0 :
            ``RState`` object with only the initial data.
            Namely, ``x_data`` and ``y_data`` are 1D.
        """
        if len(self.state.shape) == 2:
            return RState(self.state[0])
        else:
            return self


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Control:
    """Helper for indexing a control array.

    The ``control`` attribute should be 2D, with rows representing time and columns
    representing state values.
    See the various ``property`` decorators for specific indexing conventions.

    Parameters
    ----------
    control :
        Control array.
        See the properties.
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
        """X-acceleration."""
        return self.control[..., 0]

    @property
    def y(self) -> jax.Array:
        """Y-acceleration."""
        return self.control[..., 1]

    @property
    def z(self) -> jax.Array:
        """Z-acceleration."""
        return self.control[..., 2]

    @property
    def roll(self) -> jax.Array:
        """Roll acceleration."""
        return self.control[..., 3]

    @property
    def pitch(self) -> jax.Array:
        """Pitch acceleration."""
        return self.control[..., 4]

    @property
    def yaw(self) -> jax.Array:
        """Yaw acceleration."""
        return self.control[..., 5]

    @property
    def size(self) -> int:
        """Get number of time steps that the control represents.

        Returns
        -------
        n :
            Numbner of time steps, which might be ``1`` if the data only
            includes one time step (when data isn't 2D).
        """
        if len(self.control.shape) == 2:
            return self.control.shape[0]
        else:
            return 1  # 1 time step

    @classmethod
    def from_flat(cls, flat: jax.Array) -> "Control":
        """Convert a flat array to a control dataclass.

        We assume that the flat array is of the form

        .. code-block:: python

            [x0, y0, z0, roll0, pitch0, yaw0,
             x1, y1, z1, roll1, pitch1, yaw1,
             ...]

        where the first three elements are the ``x``, ``y`` and ``z``
        coordinates of the first control point, the next three are the ``roll``,
        ``pitch`` and ``yaw`` angles of the first control point, and so on for
        all control points.

        This function is useful for optimization algorithms.
        Optimization algorithms want the optimization variables to be a vector,
        for simplicity.
        So, we can't really pass 2D arrays (matrices) and dataclasses.
        But we can easily convert between these via implicit views.

        Parameters
        ----------
        flat :
            Flat array of control values.

        Returns
        -------
        control :
            ``Control`` dataclass from flat array.
        """
        assert flat.size % 6 == 0
        control = jnp.reshape(flat, (-1, 6))
        return cls(control)

    def flatten(self) -> jax.Array:
        """Get flattened controls.

        Useful for optimization algorithms, who only want to optimize over
        vectors.

        Returns
        -------
        flat :
            Robot controls as a flat array.
        """
        return jnp.ravel(self.control)

    def get0(self) -> "Control":
        """Usually create a new Control with **only** the initial control.

        Returns
        -------
        control0 :
            ``Control`` object with only the initial data.
            Namely, ``control`` is 1D.
        """
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
        refined_controls :
            Refined controls to the more granular control step.

        See Also
        --------
        :func:`exp_mpc.stewart_min.utils.control_refinement` :
            Function that actually refines the controls.
        """
        assert len(self.control.shape) == 2
        return Control(control_refinement(dt, dtp, self.control))


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TableStats:
    """Basic statistics of the solution to the MPC optimization.

    Parameters
    ----------
    time :
        Solver time.
    status :
        Solver status.
    cost :
        Minimum cost.
    """

    time: jax.Array
    status: jax.Array
    cost: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TableSol:
    """Solution to the MPC optimization.

    Parameters
    ----------
    x :
        Robot states.
    u :
        Robot control.
    vstate_irl :
        Vestibular state of the in-real-life person.
    vstate_sim :
        Vestibular state of the simulated (reference) person.
    stats :
        Basic MPC solver statistics.
    """

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
    """Get the rotation matrix.

    Parameters
    ----------
    state :
        Robot states.
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    R :
        Rotation matrix.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.rot` :
        Computes the rotation matrix.
    """
    assert state.size == 1
    return comp.rot(state.roll, state.pitch, state.yaw, use_xy)


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot_dot(state: RState, use_xy: bool = True) -> jax.Array:
    """Get the rotation matrix time derivative.

    Parameters
    ----------
    state :
        Robot states.
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    R_dot :
        Rotation matrix time derivative.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.rot_dot` :
        Computes the rotation matrix time derivative.
    """
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
    """Get the rotation matrix and its time derivative.

    Because of automatic differentiation, it is more efficient to compute both
    matrices simultaneously.

    Parameters
    ----------
    state :
        Robot states.
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    R :
        Rotation matrix.
    R_dot :
        Rotation matrix time derivative.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.rot_and_dot` :
        Computes the rotation matrix and its time derivative.
    """
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
    """Get the second derivative of the rotation matrix.

    Parameters
    ----------
    state :
        Robot states.
    control :
        Robot controls.
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    R_dot2 :
        Second derivative of the rotation matrix.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.rot_dot2` :
        Computes the second derivative of the rotation matrix.
    """
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
    """Compute leg lengths (inverse kinematics).

    Parameters
    ----------
    state :
        Robot states.
    robo_geom :
        Robot geometry.
    use_rotary :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    leg_pos :
        Leg lengths.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.leg_pos` :
        Computes the leg lengths.
    """
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
    """Compute leg velocities.

    Parameters
    ----------
    state :
        Robot states.
    robo_geom :
        Robot geometry.
    use_rotary :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    leg_vel :
        Leg velocities.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.leg_vel` :
        Computes the leg velocities.
    """
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
    """Compute leg lengths and velocities.

    Because of automatic differentiation, it is more efficient to compute both
    quantities simultaneously.

    Parameters
    ----------
    state :
        Robot states.
    robo_geom :
        Robot geometry.
    use_rotary :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    leg_pos :
        Leg lengths.
    leg_vel :
        Leg velocities.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.leg_pos_vel` :
        Computes leg lengths and leg velocities.
    """
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
    """Compute leg accelerations.

    Parameters
    ----------
    state :
        Robot states.
    control :
        Robot controls.
    robo_geom :
        Robot geometry.
    use_rotary :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    leg_acc :
        Leg accelerations.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.leg_acc` :
        Computes the leg accelerations.
    """
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
    r"""Map from Euler angle derivatives to angular velocity.

    Parameters
    ----------
    state :
        Robot states.
    world :
        ``True`` for world-frame angular velocity, and ``False`` for moving
        head-frame angular velocity.

    Returns
    -------
    PHI :
        Matrix :math:`\Phi` such that
        :math:`\omega = \Phi \, [\dot{\phi}, \dot{\theta}, \dot{\psi}]^\top`.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.transfer_PHI` :
        Computes the transfer matrix, and describes the definition of
        :math:`\Phi`.
    """
    assert state.size == 1
    return comp.transfer_PHI(state.roll, state.pitch, state.yaw, world)


@functools.partial(jax.jit, static_argnames=("world",))
def angle_vel(state: RState, world: bool = False) -> jax.Array:
    """Compute angular velocity.

    Parameters
    ----------
    state :
        Robot states.
    world :
        ``True`` for world-frame angular velocity, and ``False`` for moving
        head-frame angular velocity.

    Returns
    -------
    omega :
        Angular velocity.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.angle_vel` :
        Computes angular velocity from Euler-angle derivatives.
    """
    assert state.size == 1
    angles = [state.roll, state.pitch, state.yaw]
    angles_dot = [state.roll_dot, state.pitch_dot, state.yaw_dot]
    inputs = angles + angles_dot + [world]
    return comp.angle_vel(*inputs)


@functools.partial(jax.jit, static_argnames=("world",))
def angle_acc(
    state: RState, control: Control, world: bool = False
) -> jax.Array:
    """Compute angular acceleration.

    Parameters
    ----------
    state :
        Robot states.
    control :
        Robot controls.
    world :
        ``True`` for world-frame angular acceleration, and ``False`` for
        moving head-frame angular acceleration.

    Returns
    -------
    alpha :
        Angular acceleration.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.angle_acc` :
        Computes angular acceleration from Euler-angle derivatives and
        accelerations.
    """
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
    robo_geom: robo.RoboGeom,
    use_xy: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Compute top and bottom joint angles.

    Parameters
    ----------
    state :
        Robot states.
    robo_geom :
        Robot geometry.
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    joint_top :
        Top joint angles.
    joint_bot :
        Bottom joint angles.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.angle_joint` :
        Computes top and bottom joint angles.
    """
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
    robo_geom: robo.RoboGeom,
    use_xy: bool = True,
) -> jax.Array:
    """Compute angles at top joints.

    Parameters
    ----------
    state :
        Robot states.
    robo_geom :
        Robot geometry.
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    joint_top :
        Top joint angles.

    See Also
    --------
    :func:`exp_mpc.stewart_min.utils.angle_joint` :
        Computes both top and bottom joint angles.
    """
    assert state.size == 1
    joint_top, _ = angle_joint(state, robo_geom, use_xy)
    return joint_top


@functools.partial(jax.jit, static_argnames=["robo_geom", "use_xy"])
def angle_joint_bot(
    state: RState,
    robo_geom: robo.RoboGeom,
    use_xy: bool = True,
) -> jax.Array:
    """Compute angles at bottom joints.

    Parameters
    ----------
    state :
        Robot states.
    robo_geom :
        Robot geometry.
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    joint_bot :
        Bottom joint angles.

    See Also
    --------
    :func:`exp_mpc.stewart_min.utils.angle_joint` :
        Computes both top and bottom joint angles.
    """
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
    dt :
        Time step.
    control :
        Robot controls.
    rstate0 :
        Current robot state.
        (Not the initial state from the previous iteration.
        Namely, dissimilar to the initial states in
        :func:`exp_mpc.stewart_min.utils.get_vstate`.)

    Returns
    -------
    rstate :
        Robot states over the control horizon, including the initial state.
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

    Note that ``vspec_acc`` and ``vspec_omega`` implicitly encode the time step
    in their integration matrices.

    Parameters
    ----------
    vspec_acc :
        Vestibular specification for linear accelerations.
    vspec_omega :
        Vestibular specification for angular velocities.
    acc_ctrl :
        Control linear accelerations, for vestibular model.
        (Linear accelerations, before vestibular processing.)
    omega_ctrl :
        Control angular velocities, for vestibular model.
        (Angular velocities, before vestibular processing.)
    vstate0 :
        Initial vestibular state.

    Returns
    -------
    vstate :
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
    vspec_acc :
        Vestibular specification for linear accelerations.
    vspec_omega :
        Vestibular specification for angular velocities.
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
    vstate :
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

    Parameters
    ----------
    dt :
        Time step.
    vspec_acc :
        Vestibular specification for linear accelerations.
    vspec_omega :
        Vestibular specification for angular velocities.
    acc_ref :
        Reference linear accelerations.
    omega_ref :
        Reference angular velocities.
    rstate0 :
        Initial robot state.
    vstate0_irl :
        Initial vestibular state for in-real-life person.
        Note that ``vstate0_irl`` is one time step behind ``vstate0_sim``,
        because of feedback.
        ``control0`` is used to update ``vstate0_irl`` to the current time step.
    vstate0_sim :
        Initial vestibular state for simulated person.
    control0 :
        Initial control, i.e., the current robot acceleration.
    control :
        Robot controls.

    Returns
    -------
    rstate :
        Robot states over the control horizon, including the initial state.
    vstate_irl :
        Vestibular state of the in-real-life person.
    vstate_sim :
        Vestibular state of the simulated (reference) person.

    Warning
    -------
    The returned VState internal states should be interpreted as eigen-states.
    To get the correct internal states, you need to transform ``P @ x`` where
    the columns of ``P`` are the eigenvectors of ``E0``.
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
    """Human angular velocity.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    omega :
        The human angular velocity.
    """
    state0 = sol.x.get0()
    return angle_vel(state0)


@jax.jit
def table_angle_vel(sol: TableSol) -> jax.Array:
    """Table angular velocity.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    omega :
        The table angular velocity.
    """
    state0 = sol.x.get0()
    return angle_vel(state0, world=True)


@jax.jit
def human_angle_acc(sol: TableSol) -> jax.Array:
    """Human angular acceleration.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    alpha :
        The human angular acceleration.
    """
    state0 = sol.x.get0()
    control0 = sol.u.get0()
    return angle_acc(state0, control0)


@jax.jit
def table_angle_acc(sol: TableSol) -> jax.Array:
    """Table angular acceleration.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    alpha :
        The table angular acceleration.
    """
    state0 = sol.x.get0()
    control0 = sol.u.get0()
    return angle_acc(state0, control0, world=True)


@jax.jit
def table_angle(sol: TableSol) -> jax.Array:
    """Table angle.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    rpy :
        The table roll, pitch, and yaw angles in degrees.
    """
    state0 = sol.x.get0()
    rpy = jnp.array([state0.roll, state0.pitch, state0.yaw])
    return jnp.degrees(rpy)


@jax.jit
def table_pos(sol: TableSol) -> jax.Array:
    """Table position.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    pos :
        The table position as a 3-element array ``[x, y, z]``.
    """
    state0 = sol.x.get0()
    return jnp.array([state0.x, state0.y, state0.z])


@jax.jit
def table_vel(sol: TableSol) -> jax.Array:
    """Table velocity.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    vel :
        The table velocity as a 3-element array ``[x_dot, y_dot, z_dot]``.
    """
    state0 = sol.x.get0()
    return jnp.array([state0.x_dot, state0.y_dot, state0.z_dot])


@jax.jit
def table_acc(sol: TableSol) -> jax.Array:
    """Table acceleration.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    acc :
        The table acceleration as a 3-element array
        ``[x_dot2, y_dot2, z_dot2]``.
    """
    control0 = sol.u.get0()
    return jnp.array([control0.x, control0.y, control0.z])


@jax.jit
def human_vel(sol: TableSol) -> jax.Array:
    """Human velocity.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    vel :
        The human velocity as a 3-element array ``[x_dot, y_dot, z_dot]`` in the
        human frame.
    """
    state0 = sol.x.get0()
    R = rot(state0)
    vel = jnp.array([state0.x_dot, state0.y_dot, state0.z_dot])
    return R.T @ vel


@jax.jit
def human_acc(sol: TableSol) -> jax.Array:
    """Human acceleration.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    acc :
        The human acceleration as a 3-element array ``[x_dot2, y_dot2, z_dot2]``
        in the human frame.
    """
    state0 = sol.x.get0()
    control0 = sol.u.get0()
    acc = jnp.array([control0.x, control0.y, control0.z])
    R = rot(state0)
    return R.T @ (acc + robo.gravity)


@functools.partial(jax.jit, static_argnames=["fun"])
def _sol_vmap(
    fun: tp.Callable[[TableSol], jax.Array], sol: TableSol
) -> jax.Array:
    """vmap wrapper for TableSol functions."""
    sol.x = sol.x.pop0()  # skip initial condition
    leaves, treedef = jax.tree_util.tree_flatten(sol)

    def flat_fun(*args) -> jax.Array:
        sol = jax.tree_util.tree_unflatten(treedef, args)
        return fun(sol)

    in_axes = [0, 0] + [None] * (len(leaves) - 2)  # probably 3 `None`s
    return jax.vmap(flat_fun, in_axes)(*leaves)


@jax.jit
def human_vel_horizon(sol: TableSol) -> jax.Array:
    """Human velocity over the MPC horizon.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    vel :
        The human velocity over the MPC horizon as a sequence of 3-element
        arrays ``[x_dot, y_dot, z_dot]`` in the human frame.
    """
    return _sol_vmap(human_vel, sol)


@jax.jit
def human_angle_vel_horizon(sol: TableSol) -> jax.Array:
    """Human angular velocity over the MPC horizon.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    ang_vel :
        The human angular velocity over the MPC horizon as a sequence of
        3-element arrays ``[roll_dot, pitch_dot, yaw_dot]`` in the human frame.
    """
    return _sol_vmap(human_angle_vel, sol)


@jax.jit
def human_acc_horizon(sol: TableSol) -> jax.Array:
    """Human acceleration over the MPC horizon.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    acc :
        The human acceleration over the MPC horizon as a sequence of 3-element
        arrays ``[x_dot2, y_dot2, z_dot2]`` in the human frame.
    """
    return _sol_vmap(human_acc, sol)


@jax.jit
def human_angle_acc_horizon(sol: TableSol) -> jax.Array:
    """Human angular acceleration over the MPC horizon.

    Parameters
    ----------
    sol : TableSol
        The solution containing the state and control trajectories.

    Returns
    -------
    ang_acc :
        The human angular acceleration over the MPC horizon as a sequence of
        3-element arrays ``[roll_dot2, pitch_dot2, yaw_dot2]`` in the human
        frame.
    """
    return _sol_vmap(human_angle_acc, sol)
