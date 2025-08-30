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

import jax
import jax.numpy as jnp
import numpy as np
import exp_mpc.stewart_min.const as const
import typing as tp
import dataclasses


################
# book-keeping #
################


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Pose:
    x: jax.Array
    y: jax.Array
    z: jax.Array
    phi: jax.Array
    theta: jax.Array
    psi: jax.Array

    def xyz(self) -> jax.Array:
        return jnp.stack([self.x, self.y, self.z], axis=-1)

    def rpy(self) -> jax.Array:
        return jnp.stack([self.phi, self.theta, self.psi], axis=-1)

    def __array__(self, copy: bool = False) -> np.ndarray:
        assert type(copy) is bool
        return np.stack(
            [self.x, self.y, self.z, self.phi, self.theta, self.psi],
            axis=-1,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class PoseDot:
    x_dot: jax.Array
    y_dot: jax.Array
    z_dot: jax.Array
    phi_dot: jax.Array
    theta_dot: jax.Array
    psi_dot: jax.Array

    def xyz(self) -> jax.Array:
        return jnp.stack([self.x_dot, self.y_dot, self.z_dot], axis=-1)

    def rpy(self) -> jax.Array:
        return jnp.stack([self.phi_dot, self.theta_dot, self.psi_dot], axis=-1)

    def __array__(self, copy: bool = False) -> np.ndarray:
        assert type(copy) is bool
        vals = [self.x_dot, self.y_dot, self.z_dot]
        vals.extend([self.phi_dot, self.theta_dot, self.psi_dot])
        return np.stack(vals, axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class PoseDot2:
    x_dot2: jax.Array
    y_dot2: jax.Array
    z_dot2: jax.Array
    phi_dot2: jax.Array
    theta_dot2: jax.Array
    psi_dot2: jax.Array

    def xyz(self) -> jax.Array:
        return jnp.stack([self.x_dot2, self.y_dot2, self.z_dot2], axis=-1)

    def rpy(self) -> jax.Array:
        return jnp.stack(
            [self.phi_dot2, self.theta_dot2, self.psi_dot2], axis=-1
        )

    def __array__(self, copy: bool = False) -> np.ndarray:
        assert type(copy) is bool
        vals = [self.x_dot2, self.y_dot2, self.z_dot2]
        vals.extend([self.phi_dot2, self.theta_dot2, self.psi_dot2])
        return np.stack(vals, axis=-1)


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

    x: jax.Array
    u: jax.Array
    stats: TableStats

    def pose_at(self, i: int | jax.Array) -> Pose:
        return Pose(*self.x[..., i, :6])

    def pose_dot_at(self, i: int | jax.Array) -> PoseDot:
        return PoseDot(*self.x[..., i, 6:12])

    def pose_dot2_at(self, i: int | jax.Array) -> PoseDot2:
        return PoseDot2(*self.u[..., i, :6])

    def __iter__(self) -> tp.Iterator:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...] | jax.Array
    ) -> "TableSol":
        if isinstance(key, slice):
            x_key = slice(key.start, key.stop, key.step)
            u_key = slice(key.start, key.stop - 1, key.step)
        else:
            x_key = key
            u_key = key
        return TableSol(
            x=self.x[x_key],
            u=self.u[u_key],
            stats=self.stats,
        )


################
# jax routines #
################


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
def _get_R_and_dot(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
) -> tuple[jax.Array, jax.Array]:
    return jax.jvp(_get_R, (phi, theta, psi), (phi_dot, theta_dot, psi_dot))


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


@jax.jit
def _leg_pos(R: jax.Array, t: jax.Array) -> jax.Array:
    lengths = []
    for i in range(6):
        top_i = R @ const.tops[i] + t
        diff = top_i - const.bots[i]
        lengths.append(jnp.linalg.norm(diff))
    return jnp.array(lengths)


@jax.jit
def _leg_vel(
    R: jax.Array, t: jax.Array, R_dot: jax.Array, t_dot: jax.Array
) -> jax.Array:
    return jax.jvp(_leg_pos, (R, t), (R_dot, t_dot))[1]


@jax.jit
def _leg_pos_vel(
    R: jax.Array, t: jax.Array, R_dot: jax.Array, t_dot: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return jax.jvp(_leg_pos, (R, t), (R_dot, t_dot))


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


@functools.partial(jax.jit, static_argnames=("world",))
def _get_PHI(
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
def _angle_vel(
    phi: float,
    theta: float,
    psi: float,
    phi_dot: float,
    theta_dot: float,
    psi_dot: float,
    world: bool = False,
) -> jax.Array:
    """Angular velocity."""
    PHI = _get_PHI(phi, theta, psi, world)
    return jnp.linalg.inv(PHI) @ jnp.array([phi_dot, theta_dot, psi_dot])


@functools.partial(jax.jit, static_argnames=("world",))
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
    world: bool = False,
) -> jax.Array:
    """Angular acceleration."""
    # no product rule this time, because we already have the angular velocity
    primals = (phi, theta, psi, phi_dot, theta_dot, psi_dot)
    tangents = (phi_dot, theta_dot, psi_dot, phi_dot2, theta_dot2, psi_dot2)
    return jax.jvp(
        functools.partial(_angle_vel, world=world), primals, tangents
    )[1]


def _angle_joint(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    phi: jax.Array,
    theta: jax.Array,
    psi: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Joint angles, both top and bottom."""
    R = _get_R(phi, theta, psi)
    t = jnp.array([x, y, z])
    top_angles = []
    bot_angles = []
    for i in range(6):
        top_i = R @ const.tops[i] + t
        diff = top_i - const.bots[i]
        leg_dir = diff / jnp.linalg.norm(diff)

        top_mag = jnp.linalg.norm(jnp.cross(const.top_normals[i], leg_dir))
        top_angles.append(jnp.asin(top_mag))

        bot_mag = jnp.linalg.norm(jnp.cross(const.bot_normals[i], leg_dir))
        bot_angles.append(jnp.asin(bot_mag))

    return jnp.array(top_angles), jnp.array(bot_angles)


@jax.jit
def _angle_joint_top(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    phi: jax.Array,
    theta: jax.Array,
    psi: jax.Array,
) -> jax.Array:
    """Top joint angles."""
    return _angle_joint(x, y, z, phi, theta, psi)[0]


@jax.jit
def _angle_joint_bot(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    phi: jax.Array,
    theta: jax.Array,
    psi: jax.Array,
) -> jax.Array:
    """Bottom joint angles."""
    return _angle_joint(x, y, z, phi, theta, psi)[1]


############
# wrappers #
############


def get_R(sol: TableSol) -> jax.Array:
    """Get the rotation matrix.

    We return a jax array, because this is really a private function.
    """
    pose = sol.pose_at(1)
    return _get_R(pose.phi, pose.theta, pose.psi)


def get_R_dot(sol: TableSol) -> jax.Array:
    """Get the rotation matrix derivative."""
    pose = sol.pose_at(1)
    pose_dot = sol.pose_dot_at(1)
    return _get_R_dot(
        pose.phi,
        pose.theta,
        pose.psi,
        pose_dot.phi_dot,
        pose_dot.theta_dot,
        pose_dot.psi_dot,
    )


def get_R_dot2(sol: TableSol) -> jax.Array:
    """Get the second derivative of the rotation matrix."""
    pose = sol.pose_at(1)
    pose_dot = sol.pose_dot_at(1)
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


def leg_pos(sol: TableSol) -> jax.Array:
    """All leg lengths."""
    R = get_R(sol)
    t = sol.pose_at(1).xyz()
    return jnp.array(_leg_pos(R, t))


def leg_vel(sol: TableSol) -> jax.Array:
    """All leg velocities."""
    R = get_R(sol)
    R_dot = get_R_dot(sol)
    t = sol.pose_at(1).xyz()
    t_dot = sol.pose_dot_at(1).xyz()
    return jnp.array(_leg_vel(R, t, R_dot, t_dot))


def leg_acc(sol: TableSol) -> jax.Array:
    """All leg accelerations."""
    R = get_R(sol)
    R_dot = get_R_dot(sol)
    R_dot2 = get_R_dot2(sol)
    t = sol.pose_at(1).xyz()
    t_dot = sol.pose_dot_at(1).xyz()
    t_dot2 = sol.pose_dot2_at(0).xyz()
    return jnp.array(_leg_acc(R, t, R_dot, t_dot, R_dot2, t_dot2))


def angle_pos(sol: TableSol) -> jax.Array:
    """Angle position."""
    return sol.pose_at(1).rpy()


def human_angle_vel(sol: TableSol) -> jax.Array:
    """Human angular velocity."""
    pose = sol.pose_at(1)
    pose_dot = sol.pose_dot_at(1)
    return jnp.array(_angle_vel(*pose.rpy(), *pose_dot.rpy()))


def table_angle_vel(sol: TableSol) -> jax.Array:
    """Table angular velocity."""
    pose = sol.pose_at(1)
    pose_dot = sol.pose_dot_at(1)
    return jnp.array(_angle_vel(*pose.rpy(), *pose_dot.rpy(), world=True))


def human_angle_acc(sol: TableSol) -> jax.Array:
    """Angular acceleration."""
    pose = sol.pose_at(1)
    pose_dot = sol.pose_dot_at(1)
    pose_dot2 = sol.pose_dot2_at(0)
    return jnp.array(_angle_acc(*pose.rpy(), *pose_dot.rpy(), *pose_dot2.rpy()))


def table_angle_acc(sol: TableSol) -> jax.Array:
    """Table angular acceleration."""
    pose = sol.pose_at(1)
    pose_dot = sol.pose_dot_at(1)
    pose_dot2 = sol.pose_dot2_at(0)
    return jnp.array(
        _angle_acc(*pose.rpy(), *pose_dot.rpy(), *pose_dot2.rpy(), world=True)
    )


def table_angle(sol: TableSol) -> jax.Array:
    """Table angle."""
    pose = sol.pose_at(1)
    return jnp.degrees(pose.rpy())


def table_pos(sol: TableSol) -> jax.Array:
    """Table position."""
    pose = sol.pose_at(1)
    return jnp.array(pose.xyz())


def table_vel(sol: TableSol) -> jax.Array:
    """Table velocity."""
    pose_dot = sol.pose_dot_at(1)
    return jnp.array(pose_dot.xyz())


def table_acc(sol: TableSol) -> jax.Array:
    """Table acceleration."""
    pose_dot2 = sol.pose_dot2_at(0)
    return pose_dot2.xyz()


def angle_joint_top(sol: TableSol) -> jax.Array:
    """Top joint angles."""
    pose = sol.pose_at(1)
    return jnp.array(_angle_joint_top(*pose.xyz(), *pose.rpy()))


def angle_joint_bot(sol: TableSol) -> jax.Array:
    """Bottom joint angles."""
    pose = sol.pose_at(1)
    return jnp.array(_angle_joint_bot(*pose.xyz(), *pose.rpy()))


def human_vel(sol: TableSol) -> jax.Array:
    """Human velocity."""
    vel = sol.pose_dot_at(1).xyz()
    R = jnp.array(get_R(sol))
    R_dot = jnp.array(get_R_dot(sol))
    return R.T @ (R_dot @ const.human_displacement + vel)


def human_acc(sol: TableSol) -> jax.Array:
    """Human acceleration."""
    acc = sol.pose_dot2_at(0).xyz()
    R = jnp.array(get_R(sol))
    R_dot2 = jnp.array(get_R_dot2(sol))
    return R.T @ (R_dot2 @ const.human_displacement + acc + const.gravity)
    # return R.T @ (acc + const.gravity)
