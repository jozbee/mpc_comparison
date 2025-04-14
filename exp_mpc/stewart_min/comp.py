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

import jax
import jax.numpy as jnp
import numpy as np
import exp_mpc.stewart_min.spec as spec


@jax.jit
def _get_R(phi: float, theta: float, psi: float) -> jax.Array:
    """Get the rotation matrix."""
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
def _get_R_dot(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
) -> jax.Array:
    return jax.jvp(_get_R, (phi, theta, psi), (phi_dot, theta_dot, psi_dot))[1]


@jax.jit
def _get_R_dot2(
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
        return jax.jvp(_get_R, (phi_, theta_, psi_), tangents)[1]

    def _get_R_dot_1(
        phi_dot_: float, theta_dot_: float, psi_dot_: float
    ) -> jax.Array:
        return jax.jvp(_get_R, primals, (phi_dot_, theta_dot_, psi_dot_))[1]

    res0 = jax.jvp(_get_R_dot_0, primals, tangents)
    res1 = jax.jvp(_get_R_dot_1, tangents, tangents2)
    return res0[1] + res1[1]


def get_R(sol: spec.TableSol) -> jax.Array:
    """Get the rotation matrix.

    We return a jax array, because this is really a private function.
    """
    pose = sol.pose_at(0)
    return _get_R(pose.phi, pose.theta, pose.psi)


def get_R_dot(sol: spec.TableSol) -> jax.Array:
    """Get the rotation matrix derivative."""
    pose = sol.pose_at(0)
    pose_dot = sol.pose_dot_at(0)
    return _get_R_dot(
        pose.phi,
        pose.theta,
        pose.psi,
        pose_dot.phi_dot,
        pose_dot.theta_dot,
        pose_dot.psi_dot,
    )


def get_R_dot2(sol: spec.TableSol) -> jax.Array:
    """Get the second derivative of the rotation matrix."""
    pose = sol.pose_at(0)
    pose_dot = sol.pose_dot_at(0)
    pose_ddot = sol.pose_dot2_at(0)
    return _get_R_dot2(
        pose.phi,
        pose.theta,
        pose.psi,
        pose_dot.phi_dot,
        pose_dot.theta_dot,
        pose_dot.psi_dot,
        pose_ddot.phi_dot2,
        pose_ddot.theta_dot2,
        pose_ddot.psi_dot2,
    )


@jax.jit
def _leg_pos(R: jax.Array, t: jax.Array) -> jax.Array:
    lengths = []
    for i in range(6):
        top_i = R @ spec.tops[i] + t
        diff = top_i - spec.bots[i]
        lengths.append(jnp.linalg.norm(diff))
    return jnp.array(lengths)


@jax.jit
def _leg_vel(
    R: jax.Array, t: jax.Array, R_dot: jax.Array, t_dot: jax.Array
) -> jax.Array:
    return jax.jvp(_leg_pos, (R, t), (R_dot, t_dot))[1]


@jax.jit
def _leg_acc(
    R: jax.Array,
    t: jax.Array,
    R_dot: jax.Array,
    t_dot: jax.Array,
    R_dot2: jax.Array,
    t_dot2: jax.Array,
) -> jax.Array:
    def _leg_pos_0(R_: jax.Array, t_: jax.Array) -> jax.Array:
        return jax.jvp(_leg_pos, (R_, t_), (R_dot, t_dot))[1]

    def _leg_pos_1(R_dot_: jax.Array, t_dot_: jax.Array) -> jax.Array:
        return jax.jvp(_leg_pos, (R, t), (R_dot_, t_dot_))[1]

    res0 = jax.jvp(_leg_pos_0, (R, t), (R_dot, t_dot))[1]
    res1 = jax.jvp(_leg_pos_1, (R_dot, t_dot), (R_dot2, t_dot2))[1]
    return res0 + res1


def leg_pos(sol: spec.TableSol) -> np.ndarray:
    """All leg lengths."""
    R = get_R(sol)
    t = sol.pose_at(0).xyz()
    return np.array(_leg_pos(R, t))


def leg_vel(sol: spec.TableSol) -> np.ndarray:
    """All leg velocities."""
    R = get_R(sol)
    R_dot = get_R_dot(sol)
    t = sol.pose_at(0).xyz()
    t_dot = sol.pose_dot_at(0).xyz()
    return np.array(_leg_vel(R, t, R_dot, t_dot))


def leg_acc(sol: spec.TableSol) -> np.ndarray:
    """All leg accelerations."""
    R = get_R(sol)
    R_dot = get_R_dot(sol)
    R_dot2 = get_R_dot2(sol)
    t = sol.pose_at(0).xyz()
    t_dot = sol.pose_dot_at(0).xyz()
    t_dot2 = sol.pose_dot2_at(0).xyz()
    return np.array(_leg_acc(R, t, R_dot, t_dot, R_dot2, t_dot2))


def angle_pos(sol: spec.TableSol) -> np.ndarray:
    """Angle position."""
    return sol.pose_at(0).rpy()


@jax.jit
def _get_PHI(
    phi: float,
    theta: float,
    psi: float,  # unused, apparently...
) -> jax.Array:
    """Matrix to map table euler angle derivatives to head angular velocity."""
    # sympy generated
    return jnp.array(
        [
            [1, 0, -jnp.sin(theta)],
            [0, jnp.cos(phi), jnp.sin(phi) * jnp.cos(theta)],
            [0, -jnp.sin(phi), jnp.cos(phi) * jnp.cos(theta)],
        ]
    )


@jax.jit
def _angle_vel(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
) -> jax.Array:
    """Angular velocity."""
    PHI = _get_PHI(phi, theta, psi)
    return jnp.linalg.inv(PHI) @ jnp.array([phi_dot, theta_dot, psi_dot])


@jax.jit
def _angle_acc(
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
    """Angular acceleration."""
    # no product rule this time, because we already have the angular velocity
    primals = (phi, theta, psi, phi_dot, theta_dot, psi_dot)
    tangents = (phi_dot, theta_dot, psi_dot, phi_dot2, theta_dot2, psi_dot2)
    return jax.jvp(_angle_vel, primals, tangents)[1]


def angle_vel(sol: spec.TableSol) -> np.ndarray:
    """Angular velocity."""
    pose = sol.pose_at(0)
    pose_dot = sol.pose_dot_at(0)
    return np.array(_angle_vel(
        *pose.rpy(), *pose_dot.rpy()
    ))


def angle_acc(sol: spec.TableSol) -> np.ndarray:
    """Angular acceleration."""
    pose = sol.pose_at(0)
    pose_dot = sol.pose_dot_at(0)
    pose_dot2 = sol.pose_dot2_at(0)
    return np.array(_angle_acc(
        *pose.rpy(), *pose_dot.rpy(), *pose_dot2.rpy()
    ))


def table_acc(sol: spec.TableSol) -> np.ndarray:
    pose_dot2 = sol.pose_dot2_at(0)
    return pose_dot2.xyz()


@jax.jit
def _angle_joint(
    x: float,
    y: float,
    z: float,
    phi: float,
    theta: float,
    psi: float,
) -> tuple[jax.Array, jax.Array]:
    """Joint angles, both top and bottom."""
    R = _get_R(phi, theta, psi)
    t = jnp.array([x, y, z])
    top_angles = []
    bot_angles = []
    for i in range(6):
        top_i = R @ spec.tops[i] + t
        diff = top_i - spec.bots[i]
        leg_dir = diff / jnp.linalg.norm(diff)

        top_mag = jnp.linalg.norm(jnp.cross(spec.top_normals[i], leg_dir))
        top_angles.append(jnp.asin(top_mag))

        bot_mag = jnp.linalg.norm(jnp.cross(spec.bot_normals[i], leg_dir))
        bot_angles.append(jnp.asin(bot_mag))

    return jnp.array(top_angles), jnp.array(bot_angles)


@jax.jit
def _angle_joint_top(
    x: float,
    y: float,
    z: float,
    phi: float,
    theta: float,
    psi: float,
) -> jax.Array:
    """Top joint angles."""
    return _angle_joint(x, y, z, phi, theta, psi)[0]


@jax.jit
def _angle_joint_bot(
    x: float,
    y: float,
    z: float,
    phi: float,
    theta: float,
    psi: float,
) -> jax.Array:
    """Bottom joint angles."""
    return _angle_joint(x, y, z, phi, theta, psi)[1]


def angle_joint_top(sol: spec.TableSol) -> np.ndarray:
    """Top joint angles."""
    pose = sol.pose_at(0)
    return np.array(_angle_joint_top(
        *pose.xyz(), *pose.rpy()
    ))


def angle_joint_bot(sol: spec.TableSol) -> np.ndarray:
    """Bottom joint angles."""
    pose = sol.pose_at(0)
    return np.array(_angle_joint_bot(
        *pose.xyz(), *pose.rpy()
    ))


def human_vel(sol: spec.TableSol) -> np.ndarray:
    """Human velocity."""
    vel = sol.pose_dot_at(0).xyz()
    R_dot = np.array(get_R_dot(sol))
    return R_dot @ spec.human_displacement + vel


def human_acc(sol: spec.TableSol) -> np.ndarray:
    """Human acceleration."""
    acc = sol.pose_dot2_at(0).xyz()
    R_dot2 = np.array(get_R_dot2(sol))
    return R_dot2 @ spec.human_displacement + acc
