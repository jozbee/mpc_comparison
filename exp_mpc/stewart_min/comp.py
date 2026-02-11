"""Computations, in jax.

Routines for primitive computations, in jax.
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np

import exp_mpc.stewart_min.const as const


############
# geometry #
############


@functools.partial(jax.jit, static_argnames=["use_xy"])
def rot(phi: float, theta: float, psi: float, use_xy: bool = True) -> jax.Array:
    """Get the rotation matrix.

    (phi, theta, psi) = (roll, pitch, yaw)
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


@jax.jit
def leg_pos(R: jax.Array, t: jax.Array) -> jax.Array:
    lengths = []
    delta = const.human_displacement
    for i in range(6):  # unroll
        top_i = R @ (const.tops[i] - delta) + delta + t
        diff = top_i - const.bots[i]
        lengths.append(jnp.linalg.norm(diff))
    return jnp.array(lengths)


@jax.jit
def leg_vel(
    R: jax.Array, t: jax.Array, R_dot: jax.Array, t_dot: jax.Array
) -> jax.Array:
    return jax.jvp(leg_pos, (R, t), (R_dot, t_dot))[1]


@jax.jit
def leg_pos_vel(
    R: jax.Array, t: jax.Array, R_dot: jax.Array, t_dot: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return jax.jvp(leg_pos, (R, t), (R_dot, t_dot))


@jax.jit
def leg_acc(
    R: jax.Array,
    t: jax.Array,
    R_dot: jax.Array,
    t_dot: jax.Array,
    R_dot2: jax.Array,
    t_dot2: jax.Array,
) -> jax.Array:
    def _leg_pos_0(R_: jax.Array, t_: jax.Array) -> jax.Array:
        return jax.jvp(leg_pos, (R_, t_), (R_dot, t_dot))[1]

    def _leg_pos_1(R_dot_: jax.Array, t_dot_: jax.Array) -> jax.Array:
        return jax.jvp(leg_pos, (R, t), (R_dot_, t_dot_))[1]

    res0 = jax.jvp(_leg_pos_0, (R, t), (R_dot, t_dot))[1]
    res1 = jax.jvp(_leg_pos_1, (R_dot, t_dot), (R_dot2, t_dot2))[1]
    return res0 + res1


@functools.partial(jax.jit, static_argnames=("world",))
def transfer_PHI(
    phi: float,
    theta: float,
    psi: float,
    world: bool = False,
) -> jax.Array:
    """Matrix to map table euler angle derivatives to head angular velocity."""
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
    """Angular velocity."""
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
    """Angular acceleration."""
    # no product rule this time, because we already have the angular velocity
    primals = (phi, theta, psi, phi_dot, theta_dot, psi_dot)
    tangents = (phi_dot, theta_dot, psi_dot, phi_dot2, theta_dot2, psi_dot2)
    return jax.jvp(
        functools.partial(angle_vel, world=world), primals, tangents
    )[1]


@functools.partial(jax.jit, static_argnames=["use_xy"])
def angle_joint(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    phi: jax.Array,
    theta: jax.Array,
    psi: jax.Array,
    use_xy: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Joint angles, both top and bottom."""
    R = rot(phi, theta, psi, use_xy)
    t = jnp.array([x, y, z])
    top_angles = []
    bot_angles = []
    delta = const.human_displacement
    for i in range(6):
        top_i = R @ (const.tops[i] - delta) + delta + t
        diff = top_i - const.bots[i]
        leg_dir = diff / jnp.linalg.norm(diff)

        top_mag = jnp.linalg.norm(jnp.cross(const.top_normals[i], leg_dir))
        top_angles.append(jnp.asin(top_mag))

        bot_mag = jnp.linalg.norm(jnp.cross(const.bot_normals[i], leg_dir))
        bot_angles.append(jnp.asin(bot_mag))

    return jnp.array(top_angles), jnp.array(bot_angles)


###############
# integration #
###############


@jax.jit
def discrete_1d_euler(
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


@jax.jit
def lti_int(
    E0: jax.Array,
    E1: jax.Array,
    C: jax.Array,
    x0: jax.Array,
    u: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Fast integration scheme for SISO LTI systems.

    See `vestibular.ipynb` for more detail on the meanings.
    Namely, `E0` and `E1` implicitly encode the step size discretization, and
    `u`.
    Also, this routine is somewhat specialized for single-input single-output
    for strictly proper LTI systems.
    Note that $D = 0$ is assumed.
    (The shapes would have to be modified for the more general case.
    SISO behavior is asserted.)

    Parameters
    ----------
    E0 :
        State integration matrix.
    E1 :
        Control integration matrix.
    C :
        Observing matrix.
    x0 :
        Initial state
    u :
        Control variables.

    Returns
    -------
    A pair `(x, y)`, the internal and observed states.
    Note that `x` contains the initial state.
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

    x = jax.lax.fori_loop(0, u.size, for_body, x)
    y = jnp.squeeze(C @ x.T)
    return x, y


def lti_int_single(
    E0: jax.Array | np.ndarray,
    E1: jax.Array | np.ndarray,
    x0: jax.Array,
    u: jax.Array,
) -> jax.Array:
    """A single update of `lti_int`, but slightly more general.

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
    Single updated state
    """
    x0 = x0.reshape(-1, E0.shape[0])
    x1 = E0 @ x0.T + E1 @ u.reshape(1, -1)
    return jnp.ravel(x1.T)


def eigen_int(
    D: jax.Array | np.ndarray,
    EP1: jax.Array | np.ndarray,
    CP: jax.Array | np.ndarray,
    P_inv: jax.Array | np.ndarray,
    x0: jax.Array,
    u: jax.Array,
) -> jax.Array:
    """LTI integration, but using eigen-integration matrices.

    Note that we do not return the internal states, for efficiency.
    The observed states are all that is desired, for computation.
    If we wanted the internal states, we would also need to have access to the
    matrix `P` (or compute it from `P_inv`).

    Note that the data layout is pretty specific.
    See the beginning assertions.
    E.g., x0 is flat, but has a 2d structure that conforms (in some sense) with
    the 2d shape of u.

    Parameters
    ----------
    D :
        Eigven values for E0.
    EP1 :
        P_inv @ E1
    CP :
        C @ P
    x0 :
        Initial (internal) state.
    u :
        Controls.

    Returns
    -------
    Observed variables: y.
    """
    assert len(D.shape) == 1
    assert len(EP1.shape) == 2
    assert EP1.shape[1] == 1
    assert D.size == EP1.shape[0]
    assert P_inv.shape[0] == P_inv.shape[1] and P_inv.shape[0] == EP1.shape[0]
    assert len(x0.shape) == 1
    assert x0.size % P_inv.shape[0] == 0
    assert len(u.shape) == 2
    assert u.shape[1] == x0.size // P_inv.shape[0]
    assert u.size > 0

    # transform into the eigen vector coordinates
    x0 = jnp.transpose(P_inv @ x0.reshape(-1, D.size).T)

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

    d = jnp.tile(D, reps=(ut.shape[0], 1))
    x = jax.vmap(jax.vmap(scan_eigen_update))(x0, d, ut)
    assert isinstance(x, jax.Array)
    x = jnp.transpose(x, axes=(0, 2, 1))  # (ref, time, state)

    # desired final output shape (SISO): (ref, time)
    y = jax.vmap(lambda x: jnp.squeeze(CP @ x.T))(x)
    return y
