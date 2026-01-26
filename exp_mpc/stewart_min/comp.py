"""Computations, in jax.

Routines for primitive (mostly geometric) computations, in jax.
"""

import functools
import jax
import jax.numpy as jnp

import exp_mpc.stewart_min.const as const


@jax.jit
def rot(phi: float, theta: float, psi: float) -> jax.Array:
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
    R_z = jnp.array(
        [
            [jnp.cos(psi), -jnp.sin(psi), 0.0],
            [jnp.sin(psi), jnp.cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )  # yaw
    return R_z @ R_y @ R_x


@jax.jit
def rot_dot(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
) -> jax.Array:
    return jax.jvp(rot, (phi, theta, psi), (phi_dot, theta_dot, psi_dot))[1]


@jax.jit
def rot_and_dot(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
) -> tuple[jax.Array, jax.Array]:
    return jax.jvp(rot, (phi, theta, psi), (phi_dot, theta_dot, psi_dot))


@jax.jit
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
) -> jax.Array:
    primals = (phi, theta, psi)
    tangents = (phi_dot, theta_dot, psi_dot)
    tangents2 = (phi_dot2, theta_dot2, psi_dot2)

    # we need a product rule, so we need two jvps
    # namely, the tangents are also functions of time
    # (we have also numerically checked these implementations with sympy)

    def _get_R_dot_0(phi_: float, theta_: float, psi_: float) -> jax.Array:
        return jax.jvp(rot, (phi_, theta_, psi_), tangents)[1]

    def _get_R_dot_1(
        phi_dot_: float, theta_dot_: float, psi_dot_: float
    ) -> jax.Array:
        return jax.jvp(rot, primals, (phi_dot_, theta_dot_, psi_dot_))[1]

    res0 = jax.jvp(_get_R_dot_0, primals, tangents)
    res1 = jax.jvp(_get_R_dot_1, tangents, tangents2)
    return res0[1] + res1[1]


@jax.jit
def leg_pos(R: jax.Array, t: jax.Array) -> jax.Array:
    lengths = []
    delta = const.human_displacement
    for i in range(6):
        top_i = R @ (const.tops[i] + delta) - delta + t
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
    return jnp.linalg.inv(PHI) @ jnp.array([phi_dot, theta_dot, psi_dot])


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


@jax.jit
def angle_joint(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    phi: jax.Array,
    theta: jax.Array,
    psi: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Joint angles, both top and bottom."""
    R = rot(phi, theta, psi)
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


@jax.jit
def angle_joint_top(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    phi: jax.Array,
    theta: jax.Array,
    psi: jax.Array,
) -> jax.Array:
    """Top joint angles."""
    return angle_joint(x, y, z, phi, theta, psi)[0]


@jax.jit
def angle_joint_bot(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    phi: jax.Array,
    theta: jax.Array,
    psi: jax.Array,
) -> jax.Array:
    """Bottom joint angles."""
    return angle_joint(x, y, z, phi, theta, psi)[1]
