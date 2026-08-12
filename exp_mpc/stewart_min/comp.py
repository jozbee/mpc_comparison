"""Primitive geometry computations in JAX."""

import functools

import jax
import jax.numpy as jnp


def rot(q: jax.Array) -> jax.Array:
    r"""Rotation matrix from unit quaternion.

    We assume the scalar first convention for quaternions, i.e.,
    :math:`q = q_0 + q_1 \, i + q_2 \, j + q_3 \, k`.

    Paramters
    ---------
    q :
        Unit quaternion

    Returns
    -------
    rot :
        Rotation matrix from `q`.
    """
    assert q.shape == (4,)
    q_0, q_1, q_2, q_3 = q
    x0 = q_1**2
    x1 = q_2**2
    x2 = -x1
    x3 = q_0**2
    x4 = q_3**2
    x5 = x3 - x4
    x6 = 2 * q_0
    x7 = q_3 * x6
    x8 = q_2 * x6
    x9 = 2 * q_1
    x10 = -x0
    x11 = q_1 * x6
    return jnp.array(
        [
            [x0 + x2 + x5, 2 * q_1 * q_2 - x7, q_3 * x9 + x8],
            [q_2 * x9 + x7, x1 + x10 + x5, 2 * q_2 * q_3 - x11],
            [2 * q_1 * q_3 - x8, 2 * q_2 * q_3 + x11, x10 + x2 + x3 + x4],
        ]
    )


def rot_tilt(t: jax.Array) -> jax.Array:
    r"""Rotation matrix from tilt quaternion.

    A tilt quaternion has zero z-component, i.e.,
    :math:`t = t_0 + t_1 \, i + t_2 \, j + 0 \, k`.
    Supposing `t` is a unit quaternion, we compute the corresponding rotation
    matrix.

    Parameters
    ---------
    t :
        Tilt unit quaternion.

    Returns
    -------
    rot :
        Rotation matrix.
    """

    assert t.shape == (3,)
    t_0, t_1, t_2 = t
    x0 = 2 * t_2**2 - 1
    x1 = 2 * t_2
    x2 = t_1 * x1
    x3 = t_0 * x1
    x4 = 2 * t_1**2
    x5 = 2 * t_0 * t_1
    return jnp.array([[-x0, x2, x3], [x2, 1 - x4, -x5], [-x3, x5, -x0 - x4]])


def rot_yaw(yaw: jax.Array) -> jax.Array:
    """Rotation matrix from yaw angle.

    Parameters
    ----------
    yaw :
        Yaw angle for yaw-quaternion.

    Returns
    -------
    rot :
        Rotation matrix.
    """
    assert yaw.shape == ()
    x0 = (1 / 2) * yaw
    x1 = jnp.sin(x0)
    x2 = 1 - 2 * x1**2
    x3 = 2 * x1 * jnp.cos(x0)
    return jnp.array([[x2, -x3, 0], [x3, x2, 0], [0, 0, 1]])


def tilt2euler(t: jax.Array) -> jax.Array:
    r"""Euler angles from tilt quaternion.

    We assume the euler angle representation such that we have the rotation
    matrix decomposition :math:`R = R_z \, R_y \, R_x`.

    Parameters
    ---------
    t :
        Tilt unit quaternion.

    Returns
    -------
    euler :
        Euler angles `(x=roll, y=pitch, z=yaw)`.
    """

    assert t.shape == (3,)
    t_0, t_1, t_2 = t
    x0 = 2 * t_0
    x1 = t_1**2
    x2 = 2 * t_2**2 - 1
    x3 = 2 * x1 + x2
    return jnp.array(
        [
            jnp.arctan2(t_1 * x0, -x3),
            jnp.arctan2(t_2 * x0, jnp.sqrt(4 * t_0**2 * x1 + x3**2)),
            jnp.arctan2(2 * t_1 * t_2, -x2),
        ]
    )

def quat2euler(q):
    r"""Euler angles from unit quaternion.

    We assume the euler angle representation such that we have the rotation
    matrix decomposition :math:`R = R_z \, R_y \, R_x`.

    Parameters
    ---------
    q :
        Unit quaternion.

    Returns
    -------
    euler :
        Euler angles `(x=roll, y=pitch, z=yaw)`.
    """
    assert q.shape == (4,)
    q_0, q_1, q_2, q_3 = q
    x0 = q_0 * q_1
    x1 = q_2 * q_3
    x2 = 2 * q_2**2 - 1
    x3 = 2 * q_1**2 + x2
    x4 = 2 * q_2
    x5 = 2 * q_3
    return jnp.array(
        [
            jnp.arctan2(2 * x0 + 2 * x1, -x3),
            jnp.arctan2(
                q_0 * x4 - q_1 * x5, jnp.sqrt(x3**2 + 4 * (x0 + x1) ** 2)
            ),
            jnp.arctan2(q_0 * x5 + q_1 * x4, -2 * q_3**2 - x2),
        ]
    )


def inv_ty(q, yaw):
    r"""Get quaternion from tilt-yaw decomposition.

    The yaw vector is given by (`yaw` is a scalar)
    :math:`y = \cos(yaw / 2) + 0 \, i + 0 \, j + \sin(yaw / 2) \, k`.
    The tilt vector should be given by
    :math:`t = t_0 + t_1 \, i + t_2 \, j + 0 \, k`.
    The final quaternion is computed via the quaternion multiplication
    :math:`t \, y`.
    In practice, the tilt might have a slight nonzero component, which we
    account for.

    Parameters
    ----------
    t :
        Unit (tilt) quaternion
    yaw :
        Yaw angle.

    Returns
    -------
    quat :
        Quaternion representing the tilt-yaw composition.
    """
    assert q.shape == (4,)
    q_0, q_1, q_2, q_3 = q
    x0 = (1 / 2) * yaw
    x1 = jnp.cos(x0)
    x2 = jnp.sin(x0)
    return jnp.array(
        [
            q_0 * x1 - q_3 * x2,
            q_1 * x1 + q_2 * x2,
            -q_1 * x2 + q_2 * x1,
            q_0 * x2 + q_3 * x1,
        ]
    )


def ang_vel(q: jax.Array, c: jax.Array, dt: jax.Array) -> jax.Array:
    r"""Compute angular velocity (moving frame) from finite difference.

    The angular velocity in the moving frame is given by

    .. math::
        \dot{q} = q \, \omega

    with :math:`\omega` a vector quaternion.
    Note that :math:`\omega` gives the moving frame angular velocity.
    We approximate the derivative with a forward finite difference.

    Parameters
    ----------
    p :
        Previous unit quaternion.
    c :
        Current unit quaternion.
    dt :
        Time step.

    Returns
    -------
    omega :
        Moving frame angular velocity.
    """
    assert q.shape == (4,) and c.shape == (4,)
    q_0, q_1, q_2, q_3 = q
    c_0, c_1, c_2, c_3 = c
    x0 = 1 / dt
    return jnp.array(
        [
            x0 * (-c_0 * q_1 + c_1 * q_0 + c_2 * q_3 - c_3 * q_2),
            x0 * (-c_0 * q_2 - c_1 * q_3 + c_2 * q_0 + c_3 * q_1),
            x0 * (-c_0 * q_3 + c_1 * q_2 - c_2 * q_1 + c_3 * q_0),
        ]
    )


def fill_v(t, v):
    r"""Fill Lie Algebra vector from tilt components.

    Given a tile vector, :math:`t = t_0 + t_1 \, i + t_2 \, j + 0 \, k`, there
    is only a 2-dimensional subspace of tangent vectors :math:`V` such that
    the solution to the ODE :math:`\dot{t} = t \, v` satisfies :math:`t_3 = 0`
    for all time.
    We parameterize :math:`V` with two components, which obviously depends on
    the tilt quaternion :math:`t`.

    Parameters
    ----------
    t :
        Tilt.
    v :
        Lie Algebra vector.

    Returns
    -------
    fv :
        Filled Lie Algebra vector.
    """
    assert t.shape == (3,) and v.shape == (2,)
    v3 = (-t[1] * v[1] + t[2] * v[0]) / t[0]
    return jnp.concatenate([v, jnp.array([v3])])


@jax.jit
def _ssinc(x):
    r"""Specialized implementation of square root sinc.

    Given multidimensional input :math:`x \in \mathbb{R}^n`, we want to compute
    :math:`sin(\|x\|) / \|x\|`.
    This can be computed using the usual function `sinc`.
    However, note that the square root operation naively ruins
    differentiability when applying automatic differentiation, because the
    chain rule fails us.
    However, this function is differentiable: just look at the Taylor series
    expansion.
    Thus, we have a custom run function.
    """
    if x.size == 1:
        return jax.lax.cond(
            x <= 0.0,  # technically bad if x < 0...
            lambda: _ssinc_maclaurin(0, x),
            lambda: jnp.sin(jnp.sqrt(x)) / jnp.sqrt(x),
        )
    else:
        return jax.vmap(_ssinc)(x)


@functools.partial(jax.custom_jvp, nondiff_argnums=[0])
def _ssinc_maclaurin(k, x):
    fact = jnp.prod(jnp.arange(k + 1, 2 * k + 1 + 1, dtype=float))
    return jnp.ones_like(x) * (-1) ** k / fact


@_ssinc_maclaurin.defjvp
def _sinc_maclaurin_jvp(k, primals, tangents):
    x = primals[0]
    t = tangents[0]
    return _ssinc_maclaurin(k, x), _ssinc_maclaurin(k + 1, x) * t


@jax.jit
def _scos(x):
    """See `_ssinc`."""
    if x.size == 1:
        return jax.lax.cond(
            x <= 0,
            lambda: _scos_maclaurin(0, x),
            lambda: jnp.cos(jnp.sqrt(x)),  # usual
        )
    else:
        return jax.vmap(_scos)(x)


@functools.partial(jax.custom_jvp, nondiff_argnums=[0])
def _scos_maclaurin(k, x):
    fact = jnp.prod(jnp.arange(k + 1, 2 * k + 1, dtype=float))
    return jnp.ones_like(x) * (-1) ** k / fact


@_scos_maclaurin.defjvp
def _scos_maclaurin_jvp(k, primals, tangents):
    x = primals[0]
    t = tangents[0]
    return _scos_maclaurin(k, x), _scos_maclaurin(k + 1, x) * t


def quat_zoh(v: jax.Array, dt: jax.Array) -> jax.Array:
    r"""Compute ZoH matrix for given Lie Algebra vector.

    If :math:`q: \mathbb{R} \to \mathbb{R}^4` is a path through the unit
    quaternions, then we can write

    .. math::
        \dot{q}(t) = q(t) \, \omega(t), \qquad
        \omega = 0 + \omega_1 \, i + \omega_2 \, j + \omega_3 \, k

    where :math:`2 \, \omega` is the usual angular velocity in the moving frame.
    If :math:`\omega` is held constant over the time :math:`[0, T]`, then

    .. math::

        q(T) = e^{A \, \Delta t} \, q(0)

    where :math:`A` is the matrix represented by :math:`A \, q = q \, \omega`,
    with the RHS the usual quaternion multiplication.
    The zero-order-hold (ZoH) matrix is :math:`e^{A \, \Delta t}`.

    Parameters
    ----------
    v :
        Lie Algebra vector to update quaternion.
    dt :
        Integration time step.

    Returns
    -------
    exp :
        ZoH matrix.
    """
    assert v.shape == (3,)
    v *= dt
    v_square = jnp.sum(jnp.square(v))
    c = _scos(v_square)
    s = _ssinc(v_square)
    res = jnp.array(
        [
            [c, -v[0] * s, -v[1] * s, -v[2] * s],
            [v[0] * s, c, v[2] * s, -v[1] * s],
            [v[1] * s, -v[2] * s, c, v[0] * s],
            [v[2] * s, v[1] * s, -v[0] * s, c],
        ]
    )
    return res


def tilt_zoh(v: jax.Array, dt: jax.Array) -> jax.Array:
    r"""Compute ZoH matrix for given Lie Algebra vector for tilt quaternions.

    The only difference with `quat_zoh` is that we assume that the unit
    base point for the unit quaternion is a tilt, and v is chosen so that
    the output is also a tilt.
    Apparently this makes a performance difference in the jax implementation.

    Parameters
    ----------
    v :
        Lie Algebra vector to update quaternion.
    dt :
        Integration time step.

    Returns
    -------
    exp :
        ZoH matrix.
    """
    assert v.shape == (3,)
    v *= dt
    v_square = jnp.sum(jnp.square(v))
    c = _scos(v_square)
    s = _ssinc(v_square)
    res = jnp.array(
        [
            [c, -v[0] * s, -v[1] * s],
            [v[0] * s, c, v[2] * s],
            [v[1] * s, -v[2] * s, c],
        ]
    )
    return res


def _ssinc_deriv(x):
    r"""Derivatives of square-root sinc function.

    Given multidimensional input :math:`x \in \mathbb{R}^n`, we want to compute
    :math:`sin(\|x\|) / \|x\|`.
    This can be computed using the usual function `sinc`.
    However, note that the square root operation naively ruins
    differentiability when applying automatic differentiation, because the
    chain rule fails us.
    However, this function is differentiable: just look at the Taylor series
    expansion.
    Thus, we manually implement its derivative, and similar quantities.
    """
    sqr = jnp.sqrt(x)
    scos = jnp.cos(sqr)
    ssinc = jnp.sin(sqr) / sqr
    ssincp = (scos - ssinc) / (2 * x)
    return jax.lax.cond(
        x <= 0,
        lambda: (1.0, 1.0, -1 / 6),
        lambda: (scos, ssinc, ssincp),
    )


def _g_deriv(t, v, dt):
    """Sympy generated code differentiating the function `g`.

    See the commented line in the source code for `process_tilt_fwd` for the
    jax implementation, which is slightly slower than this sympy implementation.
    """
    assert t.shape == (3,) and v.shape == (2,)
    t_0, t_1, t_2 = t
    v_1, v_2 = v

    x0 = t_0 ** (-3)
    x1 = t_1 * v_2 - t_2 * v_1
    x2 = x1**2
    x3 = t_0**2
    x4 = dt**2
    x5 = x3 ** (-1)
    x6 = x4 * x5
    x7 = x6 * (x2 + x3 * (v_1**2 + v_2**2))

    c0, s0, s1 = _ssinc_deriv(x7)

    x8 = s0
    x9 = t_0 * x8
    x10 = t_1 * v_1 + t_2 * v_2
    x11 = s1
    x12 = 2 * x11
    x13 = dt * x12
    x14 = x10 * x13 + x9
    x15 = x2 * x4
    x16 = -x10
    x17 = x1 * x6
    x18 = x12 * x4
    x19 = x1 * x18
    x20 = -t_2 * x8 + v_1 * x19
    x21 = x12 * x15
    x22 = t_1 * x1
    x23 = dt * x9
    x24 = t_2 * x21 + x22 * x23
    x25 = -x20 * x3 + x24
    x26 = t_0**4
    x27 = dt * x1 / x26
    x28 = dt * x0
    x29 = v_2 * x28
    x30 = v_1 * x28
    x31 = t_1 * x8
    x32 = -dt * t_0 * t_2 * x1 * x8 + t_1 * x21 + x3 * (v_2 * x19 + x31)
    x33 = -x32
    x34 = t_2 * x1
    x35 = v_1 * x3 - x34
    x36 = x23 * x35
    x37 = dt * x5
    x38 = t_2 * x8
    x39 = v_2 * x3 + x22
    x40 = x23 * x39
    x41 = x18 * x34
    x42 = v_1 * x18
    x43 = -t_1 * x38
    x44 = x18 * x22
    x45 = v_2 * x18
    x46 = c0
    x47 = dt * x8
    x48 = v_1 * x47
    x49 = v_2 * x47
    x50 = x1 * x47 / t_0
    g0 = jnp.array(
        [
            [x0 * x14 * x15, v_2 * x17 * (x13 * x16 - x9), v_1 * x14 * x17],
            [x25 * x27, x29 * (x20 * x3 - x24), x25 * x30],
            [x27 * x33, x29 * x32, x30 * x33],
        ]
    )
    g1 = jnp.array(
        [
            [
                x37 * (2 * x11 * x16 * x35 * x4 - x3 * x31 - x36),
                x37 * (2 * x11 * x16 * x39 * x4 - x3 * x38 - x40),
            ],
            [
                x28
                * (
                    -t_1 * x36
                    + x26 * x8
                    + x3 * (t_2**2 * x8 + x35 * x42)
                    - x35 * x41
                ),
                x28 * (-t_1 * x40 + x3 * (x39 * x42 + x43) - x39 * x41),
            ],
            [
                x28 * (-t_2 * x36 + x3 * (x35 * x45 + x43) + x35 * x44),
                x28
                * (
                    -t_2 * x40
                    + x26 * x8
                    + x3 * (t_1**2 * x8 + x39 * x45)
                    + x39 * x44
                ),
            ],
        ]
    )
    g2 = jnp.array([[x46, -x48, -x49], [x48, x46, -x50], [x49, x50, x46]])
    return (g0, g1, g2)


def _g(x1, x2, x3, dt):
    fv = fill_v(x1, x2)
    M = tilt_zoh(fv, dt)
    return M @ x3


@jax.custom_vjp
def process_tilt(t0, v, dt):
    """Process tilt vectors from sequence of tangent vectors using ZoH.

    We implement a custom derivative rule, because the naive implementation is
    slow due to memory bounds.
    We explicitly reduce the amount of memory needed in the back propogation.

    Parameters
    ----------
    t0 :
        Initial tilt unit quaternion.
    v :
        Tangent vector for tilt quaternion (first two components).
    dt :
        Time step for zero-order-hold (ZoH) integration.

    Warning
    -------
    Our custom derivative rule does not produce meaningful outputs for
    derivatives with respect to `t0` and `dt`.
    This is not a problem for our current MPC implementation, but you are
    warned.
    """

    def comp_tilt(t, v):
        t1 = _g(t, v, t, dt)
        return t1, t1

    _, t = jax.lax.scan(comp_tilt, t0, v)
    return t


def _process_tilt_fwd(t0, v, dt):
    def comp_tilt(t, v):
        # g0, g1, g2 = jax.jacrev(_g, argnums=[0, 1, 2])(t, v, t, dt)
        g0, g1, g2 = _g_deriv(t, v, dt)
        M = g2
        t1 = M @ t
        return t1, (t1, g1, g0 + g2)

    _, (t, g1, g02) = jax.lax.scan(comp_tilt, t0, v)
    return t, (g1, g02)


def _process_tilt_bwd(res, c):
    g1, g02 = res

    def bwd_acc(a, gc):
        g1, g02, c = gc
        a = c + g02.T @ a
        w = g1.T @ a
        return a, w

    a = c[-1]
    w1 = g1[-1].T @ a
    _, w = jax.lax.scan(bwd_acc, a, (g1[:-1], g02[1:], c[:-1]), reverse=True)
    w = jnp.concatenate([w, w1.reshape(1, -1)])

    t0_dummy = jnp.zeros(3)
    dt_dummy = jnp.array(0.0)
    return t0_dummy, w, dt_dummy


process_tilt.defvjp(_process_tilt_fwd, _process_tilt_bwd)


def leg_pos(
    tops: jax.Array,
    bots: jax.Array,
) -> jax.Array:
    """Compute leg lengths (inverse kinematics).

    We assume that `tops` and `bots` are represented in the same frame.
    This is always in a fixed frame somewhere around the center of the stewart
    platform.

    Parameters
    ----------
    tops :
        Top leg coordinates, after being transformed.
    bots :
        Bottom leg coordinates.

    Returns
    -------
    ell :
        Vector of leg lengths of length 6.
    """
    ell = jnp.linalg.norm(tops - bots, axis=1)
    return ell


def leg_ang(
    tops: jax.Array,
    bots: jax.Array,
    top_normals: jax.Array,
    bot_normals: jax.Array,
) -> jax.Array:
    """Joint angles, both top and bottom.

    Parameters
    ----------
    tops :
        Top leg coordinates, after being transformed.
    bots :
        Bottom leg coordinates.
    top_normals :
        Normal vectors from tops to bots, at home.
    bot_normals :
        Normal vectors from bots to tops, at home.

    Returns
    -------
    angles :
        Top angles and bottom angles, a :math:`12` vector.
    """
    top_angles = []
    bot_angles = []
    for i in range(6):
        diff = tops[i] - bots[i]
        leg_dir = diff / jnp.linalg.norm(diff)

        top_mag = jnp.linalg.norm(jnp.cross(top_normals[i], leg_dir))
        top_angles.append(jnp.asin(top_mag))

        bot_mag = jnp.linalg.norm(jnp.cross(bot_normals[i], leg_dir))
        bot_angles.append(jnp.asin(bot_mag))

    return jnp.array(top_angles + bot_angles)


def det3(P: jax.Array) -> jax.Array:
    """Determinant of 3x3 matrix.

    Parameters
    ----------
    P :
        3x3 matrix.

    Returns
    -------
    det :
        Determinant of P.
    """
    assert P.shape == (3, 3)
    t0 = P[0, 0] * (P[1, 1] * P[2, 2] - P[1, 2] * P[2, 1])
    t1 = P[1, 0] * (P[2, 1] * P[0, 2] - P[0, 1] * P[2, 2])
    t2 = P[2, 0] * (P[0, 1] * P[1, 2] - P[0, 2] * P[1, 1])
    return t0 + t1 + t2


def inv3(P):
    """Inverse of 3x3 matrix.

    Parameters
    ----------
    P :
        3x3 matrix.

    Returns
    -------
    P_inv :
        Inverse of P.
    """
    cross = jnp.vstack(
        [
            jnp.linalg.cross(P[:, 1], P[:, 2]).reshape(1, -1),
            jnp.linalg.cross(P[:, 2], P[:, 0]).reshape(1, -1),
            jnp.linalg.cross(P[:, 0], P[:, 1]).reshape(1, -1),
        ]
    )
    return cross / det3(P)


@jax.custom_jvp
@jax.jit
def inv6(P):
    """Inverse of 6x6 matrix.

    JAX's implementation of a matrix inverse requires LAPACK by default on CPU
    (it uses LAPACK to solve several linear systems).
    This is reasonable, but my C++ exporting software doesn't link to this by
    default.
    Also, the default matrix inversion code is apparently slow:
    https://github.com/jax-ml/jax/issues/11321.
    We implement our own custom matrix inversion code (using the Schur
    complement) for :math:`6 \times 6` matrices.
    It isn't obvious that inverting the matrix should be worse than solving the
    one linear system that we are interested in, in practice.

    Parameters
    ----------
    P :
        6x6 matrix.

    Returns
    -------
    P_inv :
        Inverse of P.
    """
    assert P.shape == (6, 6)
    A = P[:3, :3]
    B = P[:3, 3:]
    C = P[3:, :3]
    D = P[3:, 3:]

    A3 = inv3(A)
    CA3 = C @ A3
    A3B = A3 @ B
    schur = inv3(D - CA3 @ B)
    schur_CA3 = schur @ CA3
    return jnp.vstack(
        [
            jnp.hstack([A3 + A3B @ schur_CA3, -A3B @ schur]),
            jnp.hstack([-schur_CA3, schur]),
        ]
    )


@inv6.defjvp
def _inv6_jvp(primals, tangents):
    # fast derivative rule
    (P,) = primals
    (P_dot,) = tangents
    P_inv = inv6(P)
    return P_inv, -P_inv @ P_dot @ P_inv


def velocity_jacobian(t: jax.Array, u: jax.Array) -> jax.Array:
    """Velocity jacobian for 6-dof Stewart platform.

    Parameters
    ----------
    t :
        Centers of top spherical joints of Stewart platform.
        Points are represented in the frame where the origin is the center
        of the Stewart platform (the moving frame).
    u :
        Unit vectors that denote the direction from the bottom spherical joints
        to the top spherical joints.
        The vectors are represented in the fixed frame.

    Returns
    -------
    vel_jac :
        The velocity jacobian.
    """
    assert t.shape == (6, 3) and u.shape == (6, 3)
    t_cross_u = jnp.linalg.cross(t, u, axis=1)
    return jnp.hstack([u, t_cross_u])


def estop_a_f(
    gravity: jax.Array,
    m: jax.Array,
    r_0: jax.Array,
    vel_jac_inv: jax.Array,
) -> jax.Array:
    """Leg length accelerations in free-fall.

    Parameters
    ----------
    gravity :
        Earth gravity 3-vector (with negative z-component).
    m :
        Ratio of mass object mass to reflected inertia from a single leg.
    r_0 :
        Center of gravity with respect to table frame (wrt platform center).
    vel_jac_inv :
        Matrix inverse of velocity jacobian for Stewart platform at the
        given time.
    """
    twist = jnp.concatenate([gravity, jnp.cross(r_0, gravity)])
    a_f = m * vel_jac_inv.T @ twist
    return a_f


def estop_delta_ell(
    t_e: jax.Array,
    a_b: jax.Array,
    leg_safety_factor: jax.Array,
    v_0: jax.Array,
    a_f: jax.Array,
) -> jax.Array:
    """Lengths after drop.

    Parameters
    ----------
    t_e :
        Time between brakes applying and e-stop being pressed.
    a_b :
        Maximum deacceleration of legs once brakes are applied.
    leg_safety_factor :
        If `delta_ell` is the leg length difference after the estop is pressed,
        then we with `leg_safety_factor * delta_ell + leg_safety`.
    v_0 :
        Initial velocity of legs, before estop.
    a_f :
        Initial acceleration.

    Returns
    -------
    delta_ell :
        Leg length change after estop is completed.
    """
    assert v_0.shape == (6,)
    assert a_f.shape == (6,)

    # note that jnp.sign(v_e) is defined to have zero gradient everywhere
    # this is the desired behavior for this function

    v_e = v_0 + a_f * t_e
    a_n = a_f - jnp.sign(v_e) * a_b
    delta_ell = v_0 * t_e + 0.5 * a_f * t_e**2 - (v_e**2) / (2 * a_n)
    return delta_ell * leg_safety_factor


def _diff(x, dt):
    return jnp.diff(x) / dt


def _taylor_coeffs(hist, dt):
    assert hist.shape == (4,)
    a0 = hist[3]
    d = functools.partial(_diff, dt=dt)
    a1 = d(hist[2:])
    a2 = d(d(hist[1:])) / 2
    a3 = d(d(d(hist))) / 6
    squee = jnp.squeeze
    return squee(a0), squee(a1), squee(a2), squee(a3)


def _make_taylor_eval(hist, dt):
    a0, a1, a2, a3 = _taylor_coeffs(hist, dt)

    def taylor_eval(x):
        return a0 + a1 * x + a2 * x**2 + a3 * x**3

    def taylor_evalp(x):
        a0, a1, a2, a3 = _taylor_coeffs(hist, dt)
        f0 = a0 + a1 * x + a2 * x**2 + a3 * x**3
        f1 = a1 + 2 * a2 * x + 3 * a3 * x**2
        f2 = 2 * a2 + 6 * a3 * x
        return jnp.array([f0, f1, f2])

    return taylor_eval, taylor_evalp


def _update_taylor(taylor_eval, dt, m, res):
    def taylor_fori(i, arr):
        t = i * dt
        arr = arr.at[i].set(taylor_eval(t))
        return arr

    res = jax.lax.fori_loop(0, m, taylor_fori, res)
    return res


def _update_lqr(x0, E, m, res):
    def lqr_fori(i, carry):
        x0, arr = carry
        x1 = E @ x0
        arr = arr.at[i].set(x1[0])
        return x1, arr

    _, res = jax.lax.fori_loop(m, res.size, lqr_fori, (x0, res))
    return res


@functools.partial(jax.jit, static_argnames=["n"])
def pred_hist(n_taylor, n, dt, E, hist):
    """Predict horizon from history.

    Parameters
    ----------
    n_taylor :
        Number of samples to take from Taylor series extrapolation.
    n :
        Total horizon to predict.
    dt :
        Time step for history.
    E :
        LQR regularizing matrix.
    hist :
        History of previous data.
        Need 4 previous data points.

    Returns
    -------
    pred :
        Predicted horizon.
    """
    taylor_eval, taylor_evalp = _make_taylor_eval(hist, dt)
    x0 = taylor_evalp(n_taylor * dt)

    res = jnp.empty(n)
    res = _update_taylor(taylor_eval, dt, n_taylor, res)
    res = _update_lqr(x0, E, n_taylor, res)
    return res
