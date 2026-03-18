"""Primitive computations in JAX.

Includes computations for the following:

* Geometry computations
* Inverse kinematics for Stewart platforms
* Integration schemes, including linear-time-invariant (LTI) systems and an
  efficient diagonal variant.

For a description of the LTI integration scheme, see the module docs for
:mod:`exp_mpc.stewart_min.vest`.
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np

import exp_mpc.stewart_min.robo as robo


############
# geometry #
############


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot(phi: float, theta: float, psi: float, use_xy: bool = True) -> jax.Array:
    r"""Get the rotation matrix specified by Euler angles.

    Parameters
    ----------
    phi :
        Roll.
    theta :
        Pitch.
    psi :
        Yaw.
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    R :
        Rotation matrix :math:`R_z \, R_y \, R_x`.
        If ``use_xy == True``, then we only return :math:`R_y \, R_x`.
    """
    R_x = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, jnp.cos(phi), -jnp.sin(phi)],
            [0.0, jnp.sin(phi), jnp.cos(phi)],
        ]
    )  # roll
    R_y = jnp.array(
        [
            [jnp.cos(theta), 0.0, jnp.sin(theta)],
            [0.0, 1.0, 0.0],
            [-jnp.sin(theta), 0.0, jnp.cos(theta)],
        ]
    )  # pitch
    if not use_xy:
        R_z = jnp.array(
            [
                [jnp.cos(psi), -jnp.sin(psi), 0.0],
                [jnp.sin(psi), jnp.cos(psi), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )  # yaw
        return R_z @ R_y @ R_x
    else:
        return R_y @ R_x


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot_dot(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
    use_xy: bool = True,
) -> jax.Array:
    r"""Get the time derivative of a rotation matrix specified by Euler angles.

    Parameters
    ----------
    phi :
        Roll.
    theta :
        Pitch.
    psi :
        Yaw.
    phi_dot :
        Roll derivative.
    theta_dot :
        Pitch derivative.
    psi_dot :
        Yaw derivative
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    R_dot :
        Rotation matrix derivative
        :math:`\frac{\mathrm{d}}{\mathrm{d} t} (R_z \, R_y \, R_x)`.
        If ``use_xy == True``, then we only return
        :math:`\frac{\mathrm{d}}{\mathrm{d} t} (R_y \, R_x)`.
    """
    rot_part = functools.partial(rot, use_xy=use_xy)
    return jax.jvp(rot_part, (phi, theta, psi), (phi_dot, theta_dot, psi_dot))[
        1
    ]


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot_and_dot(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
    use_xy: bool = True,
) -> tuple[jax.Array, jax.Array]:
    r"""Get the rotation matrix and its time derivative, from Euler angles.

    Because this function uses automatic differentiation, it is more efficient
    to call this function compared to separately computing the rotation and its
    time derivative.

    Parameters
    ----------
    phi :
        Roll.
    theta :
        Pitch.
    psi :
        Yaw.
    phi_dot :
        Roll derivative.
    theta_dot :
        Pitch derivative.
    psi_dot :
        Yaw derivative
    use_xy :
        ``True`` to ignore yaw, and ``False`` to use all Euler angles.

    Returns
    -------
    R :
        Rotation matrix :math:`R_z \, R_y \, R_x`.
        If ``use_xy == True``, then we only return :math:`R_y \, R_x`.
    R_dot :
        Rotation matrix derivative
        :math:`\frac{\mathrm{d}}{\mathrm{d} t} (R_z \, R_y \, R_x)`.
        If ``use_xy == True``, then we only return
        :math:`\frac{\mathrm{d}}{\mathrm{d} t} (R_y \, R_x)`.
    """
    rot_part = functools.partial(rot, use_xy=use_xy)
    return jax.jvp(rot_part, (phi, theta, psi), (phi_dot, theta_dot, psi_dot))


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot_dot2(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
    phi_dot2: float,
    theta_dot2: float,
    psi_dot2: float,
    use_xy: bool = True,
) -> jax.Array:
    r"""Get the second time derivative of a rotation matrix, from Euler angles.

    Parameters
    ----------
    phi :
        Roll.
    theta :
        Pitch.
    psi :
        Yaw.
    phi_dot :
        Roll derivative.
    theta_dot :
        Pitch derivative.
    psi_dot :
        Yaw derivative
    phi_dot2 :
        Second roll derivative.
    theta_dot2 :
        Second pitch derivative.
    psi_dot2 :
        Second yaw derivative
    use_xy :
        True to ignore yaw, and False to use all Euler angles.

    Returns
    -------
    R_dot :
        Rotation matrix derivative
        :math:`\frac{\mathrm{d}^2}{\mathrm{d} t^2} (R_z \, R_y \, R_x)`.
        If ``use_xy == True``, then we only return
        :math:`\frac{\mathrm{d}^2}{\mathrm{d} t^2} (R_y \, R_x)`.
    """
    rot_part = functools.partial(rot, use_xy=use_xy)
    primals = (phi, theta, psi)
    tangents = (phi_dot, theta_dot, psi_dot)
    tangents2 = (phi_dot2, theta_dot2, psi_dot2)

    # we need a product rule, so we need two jvps
    # namely, the tangents are also functions of time
    # (we have also numerically checked these implementations with sympy)

    def _get_R_dot_0(phi_: float, theta_: float, psi_: float) -> jax.Array:
        return jax.jvp(rot_part, (phi_, theta_, psi_), tangents)[1]

    def _get_R_dot_1(
        phi_dot_: float, theta_dot_: float, psi_dot_: float
    ) -> jax.Array:
        return jax.jvp(rot_part, primals, (phi_dot_, theta_dot_, psi_dot_))[1]

    res0 = jax.jvp(_get_R_dot_0, primals, tangents)
    res1 = jax.jvp(_get_R_dot_1, tangents, tangents2)
    return res0[1] + res1[1]


@functools.partial(jax.jit, static_argnames=["geom"])
def leg_pos(geom: robo.RoboGeom, R: jax.Array, t: jax.Array) -> jax.Array:
    """Compute leg lengths (inverse kinematics).

    Parameters
    ----------
    geom :
        Robot geometry.
    R :
        Rotation matrix.
    t :
        Translation vector.

    Returns
    -------
    ell :
        Vector of leg lengths of shape :math:`6 \times 3`.
    """
    lengths = []
    delta = geom.human_displacement
    for i in range(6):  # unroll
        top_i = R @ (geom.tops[i] - delta) + delta + t
        diff = top_i - geom.bots[i]
        lengths.append(jnp.linalg.norm(diff))
    return jnp.array(lengths)


@functools.partial(jax.jit, static_argnames=["geom"])
def leg_vel(
    geom: robo.RoboGeom,
    R: jax.Array,
    t: jax.Array,
    R_dot: jax.Array,
    t_dot: jax.Array,
) -> jax.Array:
    """Compute leg length velocities (inverse kinematics).

    Parameters
    ----------
    geom :
        Robot geometry.
    R :
        Rotation matrix.
    t :
        Translation vector.
    R_dot :
        Rotation matrix time derivative.
    t_dot :
        Translation vector time derivative.

    Returns
    -------
    ell_dot :
        Vector of leg velcoties of shape :math:`6 \times 3`.
    """
    leg_pos_geom = functools.partial(leg_pos, geom)
    return jax.jvp(leg_pos_geom, (R, t), (R_dot, t_dot))[1]


@functools.partial(jax.jit, static_argnames=["geom"])
def leg_pos_vel(
    geom: robo.RoboGeom,
    R: jax.Array,
    t: jax.Array,
    R_dot: jax.Array,
    t_dot: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute leg lengths and their velocities.

    Parameters
    ----------
    geom :
        Robot geometry.
    R :
        Rotation matrix.
    t :
        Translation vector.
    R_dot :
        Rotation matrix time derivative.
    t_dot :
        Translation vector time derivative.

    Returns
    -------
    ell :
        Vector of leg lengths of shape :math:`6 \times 3`.
    ell_dot :
        Vector of leg velcoties of shape :math:`6 \times 3`.
    """
    leg_pos_geom = functools.partial(leg_pos, geom)
    return jax.jvp(leg_pos_geom, (R, t), (R_dot, t_dot))


@functools.partial(jax.jit, static_argnames=["geom"])
def leg_acc(
    geom: robo.RoboGeom,
    R: jax.Array,
    t: jax.Array,
    R_dot: jax.Array,
    t_dot: jax.Array,
    R_dot2: jax.Array,
    t_dot2: jax.Array,
) -> jax.Array:
    """Compute leg length accelerations.

    Parameters
    ----------
    geom :
        Robot geometry.
    R :
        Rotation matrix.
    t :
        Translation vector.
    R_dot :
        Rotation matrix time derivative.
    t_dot :
        Translation vector time derivative.
    R_dot2 :
        Rotation matrix second time derivative.
    t_dot2 :
        Translation vector second time derivative.

    Returns
    -------
    ell_dot2 :
        Vector of leg accelerations of shape :math:`6 \times 3`.
    """
    leg_pos_geom = functools.partial(leg_pos, geom)

    def leg_pos_0(R_: jax.Array, t_: jax.Array) -> jax.Array:
        return jax.jvp(leg_pos_geom, (R_, t_), (R_dot, t_dot))[1]

    def leg_pos_1(R_dot_: jax.Array, t_dot_: jax.Array) -> jax.Array:
        return jax.jvp(leg_pos_geom, (R, t), (R_dot_, t_dot_))[1]

    res0 = jax.jvp(leg_pos_0, (R, t), (R_dot, t_dot))[1]
    res1 = jax.jvp(leg_pos_1, (R_dot, t_dot), (R_dot2, t_dot2))[1]
    return res0 + res1


@functools.partial(jax.jit, static_argnames=("world",))
def transfer_PHI(
    phi: float,
    theta: float,
    psi: float,
    world: bool = False,
) -> jax.Array:
    r"""Matrix to map table euler angle derivatives to head angular velocity.

    Parameters
    ----------
    phi :
        Roll.
    theta :
        Pitch.
    psi :
        Yaw.
    world :
        ``True`` to get transfer matrix for world frame, and ``False`` for the
        moving head frame.

    Returns
    -------
    PHI :
        Matrix :math:`\Phi` such that the angular velocity is computed via
        the vector matrix product
        :math:`\omega = \Phi \, [\dot{\phi}, \dot{\theta}, \dot{\psi}]^\top`.

    Notes
    -----
    Let the subscripts :math:`\mathrm{w}`, :math:`\mathrm{h}`, and
    :math:`\mathrm{r}` denote world frame, head frame, and robot frame,
    respectively.
    Suppose that we have an :math:`\mathrm{SE}(3)` transformation
    :math:`(R, \Delta)` such that

    .. math::

        x_{\mathrm{w}} = R \, x_{\mathrm{h}} + \Delta.

    If :math:`x_{\mathrm{h}}` is a fixed point in the head frame, i.e., if
    :math:`\dot{x}_{\mathrm{h}} = 0`, then

    .. math::

        \dot{x}_{\mathrm{w}} &= \dot{R} \, x_{\mathrm{h}} + \dot{\Delta} \\
        &= \dot{R} \, R^\top \, (x_{\mathrm{w}} - \Delta) + \dot{\Delta}.

    Because :math:`\dot{R} \, R^\top` is skew symmetric (from the identity
    :math:`R \, R^\top \equiv I`), we can define the world-frame angular
    velocity :math:`\omega_{\mathrm{w}} \in \mathbb{R}^3` to be the vector
    :math:`[\omega]_\times = \dot{R} \, R^\top`, where
    
    .. math::

        [\omega]_\times =
        \begin{bmatrix}
            0 & -\omega_z & \omega_y \\
            \omega_z & 0 & -\omega_x \\
            -\omega_y & \omega_x & 0
        \end{bmatrix},

    which parameterizes the skew symmetric matrices.
    To transform this vector into the head frame, we have
    :math:`\omega_{\mathrm{h}} = R^\top \, \omega_{\mathrm{w}}`.
    The vector algebra identity
    :math:`u \cdot (v \times w) = v \cdot (w \times u)` gives
    :math:`R \, [\omega]_\times R^\top = [R \, \omega]_\times` so that
    :math:`\omega_{\mathrm{h}} = R^\top \, \dot{R}`.
    Using the chain rule, we can compute the angular velocity matrices, and
    we can compute the linear map :math:`\Phi` that computes
    :math:`\omega = \Phi \, [\dot{\phi}, \dot{\theta}, \dot{\psi}]^\top`.
    This is pretty easily computed via
    `SymPy <https://github.com/sympy/sympy>`__.
    Example code follows

    .. code-block:: python

        def make_R() -> sp.Matrix:
            R_x = sp.Matrix(
                [
                    [1, 0, 0],
                    [0, sp.cos(phi), -sp.sin(phi)],
                    [0, sp.sin(phi), sp.cos(phi)],
                ]
            )  # roll
            R_y = sp.Matrix(
                [
                    [sp.cos(theta), 0, sp.sin(theta)],
                    [0, 1, 0],
                    [-sp.sin(theta), 0, sp.cos(theta)],
                ]
            )  # pitch
            R_z = sp.Matrix(
                [
                    [sp.cos(psi), -sp.sin(psi), 0],
                    [sp.sin(psi), sp.cos(psi), 0],
                    [0, 0, 1],
                ]
            )  # yaw
            return R_z * R_y * R_x  # type: ignore

        R = make_R()
        R

        def make_omega_vec(world=False):
            if world:
                # table angular velocity (wrt world)
                omega_mat = sp.simplify(R.diff(t) * R.T)
            else:
                # head angular velocity (wrt instantaneous table)
                omega_mat = sp.simplify(R.T * R.diff(t))

            omega_vec = sp.Matrix([omega_mat[2, 1], omega_mat[0, 2], omega_mat[1, 0]])
            omega_vec = sp.simplify(omega_vec)
            
            return omega_vec

        def make_PHI(world=False):
            if world:
                # table angular velocity (wrt world)
                omega_mat = sp.simplify(R.diff(t) * R.T)
            else:
                # head angular velocity (wrt instantaneous table)
                omega_mat = sp.simplify(R.T * R.diff(t))

            omega_vec = sp.Matrix([omega_mat[2, 1], omega_mat[0, 2], omega_mat[1, 0]])
            omega_vec = sp.simplify(omega_vec)

            euler_diff = [phi.diff(t), theta.diff(t), psi.diff(t)]
            PHI = sp.Matrix(3, 3, lambda i, j: omega_vec[i].coeff(euler_diff[j]))
            PHI = sp.simplify(PHI)
            
            return PHI

        PHI = make_PHI()
        PHI

    Finally, we provide a more intuitive interpretation of angular velocity in
    the head frame.
    This is for the author's personal reference.
    Take the current time to be :math:`t = 0`, and consider the robot frame

    .. math::

        x_{\mathrm{r}} = R_{\mathrm{h}} \, x_{\mathrm{h}} + \Delta_{\mathrm{h}}.

    where
    
    .. math::

        R(t) &=: R(0) \, R_{\mathrm{h}}(t) \\
        \Delta(t) &=: \Delta(0) + R(0) \, \Delta_h(t).

    Then, as before, we have

    .. math::

        \dot{x}_{\mathrm{r}} = \dot{R}_{\mathrm{h}} \, R_{\mathrm{h}}^\top \,
        (x_{\mathrm{r}} - \Delta_{\mathrm{h}}) + \dot{\Delta}_{\mathrm{h}}.

    At :math:`t = 0`, then :math:`R_{\mathrm{h}} = R^\top \, R = I` and

    .. math::

        \dot{R}_{\mathrm{h}} \, R_{\mathrm{h}}^\top = (R^\top \, \dot{R}) \, I)
        = R^\top \, \dot{R}.

    So, the angular velocity in the head frame an be interpreted as the
    infitesimal world-frame angular velocity.
    """
    # sympy generated
    if not world:
        return jnp.array(
            [
                [1, 0, -jnp.sin(theta)],
                [0, jnp.cos(phi), jnp.sin(phi) * jnp.cos(theta)],
                [0, -jnp.sin(phi), jnp.cos(phi) * jnp.cos(theta)],
            ]
        )
    else:
        return jnp.array(
            [
                [jnp.cos(psi) * jnp.cos(theta), -jnp.sin(psi), 0],
                [jnp.sin(psi) * jnp.cos(theta), jnp.cos(psi), 0],
                [-jnp.sin(theta), 0, 1],
            ]
        )


@functools.partial(jax.jit, static_argnames=("world",))
def angle_vel(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
    world: bool = False,
) -> jax.Array:
    """Compute Angular velocity vector.

    Parameters
    ----------
    phi :
        Roll.
    theta :
        Pitch.
    psi :
        Yaw.
    phi_dot :
        Roll velocity.
    theta_dot :
        Pitch velocity.
    psi_dot :
        Yaw velocity.
    world :
        True to compute angular velocity in the world frame, and False for the
        head (moving) frame.

    Returns
    -------
    omega :
        Angular velocity.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.transfer_PHI` :
        Transfer matric to compute angular velocity.
    """
    PHI = transfer_PHI(phi, theta, psi, world)
    return PHI @ jnp.array([phi_dot, theta_dot, psi_dot])


@functools.partial(jax.jit, static_argnames=("world",))
def angle_acc(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
    phi_dot2: float,
    theta_dot2: float,
    psi_dot2: float,
    world: bool = False,
) -> jax.Array:
    """Compute Angular acceleration vector.

    Parameters
    ----------
    phi :
        Roll.
    theta :
        Pitch.
    psi :
        Yaw.
    phi_dot :
        Roll velocity.
    theta_dot :
        Pitch velocity.
    psi_dot :
        Yaw velocity.
    phi_dot2 :
        Roll acceleration.
    theta_dot2 :
        Pitch acceleration.
    psi_dot2 :
        Yaw acceleration.
    world :
        True to compute angular acceleration in the world frame, and False for
        the head (moving) frame.

    Returns
    -------
    alpha :
        Angular acceleration.

    See Also
    --------
    :func:`exp_mpc.stewart_min.comp.transfer_PHI` :
        Transfer matric to compute angular velocity.
    :func:`jax.jvp` :
        For automatic differentiation.
    """
    # no product rule this time, because we already have the angular velocity
    primals = (phi, theta, psi, phi_dot, theta_dot, psi_dot)
    tangents = (phi_dot, theta_dot, psi_dot, phi_dot2, theta_dot2, psi_dot2)
    return jax.jvp(
        functools.partial(angle_vel, world=world), primals, tangents
    )[1]


@functools.partial(jax.jit, static_argnames=["geom", "use_xy"])
def angle_joint(
    geom: robo.RoboGeom,
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    phi: jax.Array,
    theta: jax.Array,
    psi: jax.Array,
    use_xy: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Joint angles, both top and bottom.

    Parameters
    ----------
    geom :
        Robot geometry, for leg coordinates.
    x :
        X-translation.
    y :
        Y-translation.
    z :
        Z-translation.
    phi :
        Roll.
    theta :
        Pitch.
    psi :
        Yaw.
    use_xy :
        True to have yaw only be used in the rotary table top.
        False to have the table base yaw.

    Returns
    -------
    top_angles :
        Top angles, a :math:`6` vector.
    bot_angles :
        Bottom angles, a :math:`6` vector.
    """
    R = rot(phi, theta, psi, use_xy)
    t = jnp.array([x, y, z])
    top_angles = []
    bot_angles = []
    delta = geom.human_displacement
    for i in range(6):
        top_i = R @ (geom.tops[i] - delta) + delta + t
        diff = top_i - geom.bots[i]
        leg_dir = diff / jnp.linalg.norm(diff)

        top_mag = jnp.linalg.norm(jnp.cross(geom.top_normals[i], leg_dir))
        top_angles.append(jnp.asin(top_mag))

        bot_mag = jnp.linalg.norm(jnp.cross(geom.bot_normals[i], leg_dir))
        bot_angles.append(jnp.asin(bot_mag))

    return jnp.array(top_angles), jnp.array(bot_angles)


###############
# integration #
###############


@jax.jit
def discrete_1d_euler(
    dt: jax.Array,
    x0: jax.Array,
    v0: jax.Array,
    a: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Discrete 1D Euler integration, with scalar initial data.

    Simply compute the double integrator
    
    .. math::

        v_{k + 1} &= v_k + (\Delta t) \, a_k. \\
        x_{k + 1} &= x_k + (\Delta t) \, v_k + \frac{1}{2} \, (\Delta t)^2 \,
        a_k.

    Parameters
    ----------
    dt :
        Uniform time step.
    x0 :
        Initial position.
    v0 :
        Initial velocity.
    a :
        Constant accelerations.

    Returns
    -------
    x :
        Integrated position.
    v :
        Integrated velocity.
    """
    a = jnp.ravel(a)  # really, an assertion...
    v = jnp.cumsum(jnp.concatenate([jnp.array([v0]), dt * a]))
    v0 = dt * v[:-1] + 0.5 * dt**2 * a
    x = jnp.cumsum(jnp.concatenate([jnp.array([x0]), v0]))
    return x, v


@jax.jit
def lti_int(
    E0: jax.Array,
    E1: jax.Array,
    C: jax.Array,
    D: jax.Array,
    x0: jax.Array,
    u: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Fast integration scheme for SISO LTI systems.

    See the module docs for :mod:`exp_mpc.stewart_min.vest` for mathematical
    interpretations.
    Namely, ``E0`` and ``E1`` implicitly encode the step size discretization, and
    `u`.
    Also, this routine is somewhat specialized for SISO LTI systems.

    Parameters
    ----------
    E0 :
        State integration matrix.
    E1 :
        Control integration matrix.
    C :
        :math:`y = C \, x + D \, u`.
    D :
        :math:`y = C \, x + D \, u`.
    x0 :
        Initial state
    u :
        Control variables.

    Returns
    -------
    x :
        Internal states.
        Also contains the initial state.
    y :
        Observed states.
    """
    x0 = jnp.ravel(x0)
    u = jnp.ravel(u)
    E1 = jnp.squeeze(E1)
    C = jnp.squeeze(C)

    assert E0.shape[0] == E0.shape[1]
    assert len(E1.shape) == 1
    assert E1.size == E0.shape[1]
    assert len(C.shape) == 1
    assert C.size == E0.shape[1]
    assert x0.size == E0.shape[1]
    assert u.size > 0

    x = jnp.empty(shape=(u.size + 1, x0.size), dtype=float)
    x = x.at[0].set(x0)

    def for_body(i: int, x: jax.Array) -> jax.Array:
        xi = E0 @ x[i] + E1 * u[i]
        x = x.at[i + 1].set(xi)
        return x

    # skip intial condition, because can't optimize over it, and it isn't
    #  really meaninful in the context of having a matrix D
    x = jax.lax.fori_loop(0, u.size, for_body, x)
    x = x[1:]
    y = C @ x.T + jnp.squeeze(D @ jnp.atleast_2d(u))
    return x, y


def lti_int_single(
    E0: jax.Array | np.ndarray,
    E1: jax.Array | np.ndarray,
    x0: jax.Array,
    u: jax.Array,
) -> jax.Array:
    """A single update of :func:`exp_mpc.stewart_min.comp.lti_int`.

    Technically, because of shapes, this is slightly more general than
    :func:`exp_mpc.stewart_min.comp.lti_int`.

    Parameters
    ----------
    E0 :
        State integration matrix
    E1 :
        Control integration matrix
    x0 :
        Initial state
    u :
        Control on x0.

    Returns
    -------
    x1 :
        Single updated state.
    """
    x0 = x0.reshape(-1, E0.shape[0])
    x1 = E0 @ x0.T + E1 @ u.reshape(1, -1)
    return jnp.ravel(x1.T)


def eigen_int(
    eig: jax.Array | np.ndarray,
    EP1: jax.Array | np.ndarray,
    CP: jax.Array | np.ndarray,
    D: jax.Array | np.ndarray,
    P_inv: jax.Array | np.ndarray,
    x0: jax.Array,
    u: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """LTI integration, but using eigen-integration matrices.

    Note that we do not return the internal states, for efficiency.
    The observed states are all that is desired, for computation.
    If we wanted the internal states, we would also need to have access to the
    matrix ``P`` (or compute it from ``P_inv``).

    Note that the data layout is pretty specific.
    See the beginning assertions.
    E.g., ``x0`` is flat, but has a 2d structure that conforms (in some sense) with
    the 2d shape of ``u``.

    Parameters
    ----------
    eig :
        Eigven values for ``E0``.
    EP1 :
        ``P_inv @ E1``.
    CP :
        ``C @ P``.
    D :
        ``y = C @ x + D @ u``.
    P_inv :
        Matrix inverse of ``P``.
    x0 :
        Initial (internal) state.
    u :
        Controls.

    Returns
    -------
    x :
        Internal states.
    y :
        Observed states.

    See Also
    --------
    :mod:`exp_mpc.stewart_min.vest` :
        Specifies the meaning of the integration matrices and the faster eigen
        implementation.

    Warning
    -------
    The returned internal states ``x`` are in the eigen-basis, so to get the
    actual states, one needs to transform to ``P @ x``.
    """
    assert len(eig.shape) == 1
    assert len(EP1.shape) == 2
    assert EP1.shape[1] == 1
    assert eig.size == EP1.shape[0]
    assert P_inv.shape[0] == P_inv.shape[1] and P_inv.shape[0] == EP1.shape[0]
    assert len(x0.shape) == 1
    assert x0.size % P_inv.shape[0] == 0
    assert len(u.shape) == 2
    assert u.shape[1] == x0.size // P_inv.shape[0]
    assert u.size > 0

    # transform into the eigen vector coordinates
    x0 = jnp.transpose(P_inv @ x0.reshape(-1, eig.size).T)

    # control wizardry with shapes...
    # ut is a 3-tensor, with indices
    #  (references index, state component index, time index)
    ut = jnp.transpose(EP1 @ u.T.reshape(1, -1))  # (time, ref/state)
    ut = ut.reshape(x0.shape[0], -1, x0.shape[1])  # (ref, time, state)
    ut = jnp.transpose(ut, axes=(0, 2, 1))  # (ref, state, time)

    # desired final state shape: (ref, time, state)
    # iterations:
    #  initial conditions, components, time

    def eigen_update(d, x0, u):
        x1 = d * x0 + u
        return x1, x1

    def scan_eigen_update(x0, d, u):
        part_eigen_update = functools.partial(eigen_update, d)
        return jax.lax.scan(part_eigen_update, x0, u)[1]

    d = jnp.tile(eig, reps=(ut.shape[0], 1))
    x = jax.vmap(jax.vmap(scan_eigen_update))(x0, d, ut)
    assert isinstance(x, jax.Array)
    x = jnp.transpose(x, axes=(0, 2, 1))  # (ref, time, state)

    def get_y(x, u):
        return jnp.squeeze(CP @ x.T + D @ jnp.atleast_2d(u))

    # desired final output shape (SISO): (time, ref)
    y = jnp.transpose(jax.vmap(get_y)(x, u.T))
    x = jnp.transpose(x, axes=(1, 0, 2))  # (time, ref, state)
    x = x.reshape(x.shape[0], -1)  # (time, flattened ref-states)
    return x, y
