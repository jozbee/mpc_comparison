"""Routines for computing useful MPC quantities."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from exp_mpc.stewart_min import comp, mpc_spec, siso


def prefilt_u(
    spec: mpc_spec.MPCSpec,
    u: jax.Array,
    prefilt0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Prefilter controls.

    Pre filters for

    Parameters
    ----------
    spec :
        MPC specification.
    u :
        MPC controls (additive).
    prefilt0 :
        Initial filter conditions for MPC controls.
        Includes intial tilt.

    Returns
    -------
    x :
        Internal states.
    y :
        Controls to apply to robots.
    """
    assert len(u.shape) == 1 and u.size % 6 == 0
    u = u.reshape(-1, 6)
    u_xyzyaw = u[:, :4]
    u_tilt = u[:, 4:]

    n = spec.ctrlspec.n_state
    assert len(prefilt0.shape) == 1 and prefilt0.size == 6 * n + 3
    filt0_xyzyaw = prefilt0[: n * 4].reshape(-1, n)
    filt0_tilt = prefilt0[n * 4 : n * 6]
    tilt0 = prefilt0[n * 6 :]
    prefilt0 = prefilt0[: n * 6]

    assert spec.ctrlspec.D == 0.0  # static
    y0 = spec.ctrlspec.C @ prefilt0.reshape(-1, n).T
    y0_xyzyaw = y0[:4]
    y0_tilt = y0[4:]

    ABCD = [spec.ctrlspec.A, spec.ctrlspec.B, spec.ctrlspec.C, spec.ctrlspec.D]
    add_lti_int = functools.partial(siso.additive_lti_int, *ABCD)
    tilt_lti_int = functools.partial(siso.vec_tilt_quat_lti_int, *ABCD)

    x_xyzyaw, _, y_xyzyaw = jax.vmap(add_lti_int)(
        filt0_xyzyaw, y0_xyzyaw, u_xyzyaw.T
    )
    x_tilt, y_tilt = tilt_lti_int(filt0_tilt, y0_tilt, tilt0, u_tilt, spec.dt)

    x_xyzyaw = jnp.transpose(x_xyzyaw, axes=[1, 0, 2])
    x_xyzyaw = jnp.reshape(x_xyzyaw, shape=(x_xyzyaw.shape[0], -1))

    x = jnp.hstack([x_xyzyaw, x_tilt])
    y = jnp.hstack([y_xyzyaw.T, y_tilt])
    return x, y


def apply_u(
    spec: mpc_spec.MPCSpec,
    u: jax.Array,
    filt0: jax.Array,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    """Apply control to current robot state.

    Parameters
    ----------
    spec :
        MPC specification.
    u :
        Controls
    filt0 :
        Initial filter values, as a flat array that needs to be partitioned
        with respect to "understood" conventions.

    Returns
    -------
    x_xyz :
        Internal states for linear translation.
    y_xyz :
        Observed states for linear translation.
    x_yaw :
        Internal states for yaw angle.
    y_yaw :
        Observates states for yaw angle.
    x_quat :
        Internal states for quat.
    y_quat :
        Observed states for quat.
    """
    # setup
    state0_size = (
        3 * spec.xyzspec.n_state + spec.yspec.n_state + 4 * spec.qspec.n_state
    )
    assert filt0.shape == (state0_size,)
    assert len(u.shape) == 2 and u.shape[1] == 7

    # partition
    acc = 0
    xyz0 = filt0[acc : acc + 3 * spec.xyzspec.n_state].reshape(3, -1)
    acc += 3 * spec.xyzspec.n_state
    yaw0 = filt0[acc : acc + spec.yspec.n_state]
    acc += spec.yspec.n_state
    quat0 = filt0[acc : acc + 4 * spec.qspec.n_state]

    u_xyz = u[:, :3].reshape(-1, 3)
    u_yaw = u[:, 3]
    u_tilt = u[:, 4:].reshape(-1, 3)

    # apply zero control to z-component of quaternion
    u_quat = jnp.hstack([u_tilt, jnp.zeros((u_tilt.shape[0], 1))])

    # compute
    def ss_terms(ss):
        return ss.A, ss.B, ss.C, ss.D

    x_xyz, y_xyz = jax.vmap(
        lambda x0, u: siso.lti_int(*ss_terms(spec.xyzspec), x0, u),
        in_axes=[0, 1],
    )(xyz0, u_xyz)
    x_yaw, y_yaw = siso.lti_int(*ss_terms(spec.yspec), yaw0, u_yaw)
    x_quat, y_quat = siso.quat_lti_int(*ss_terms(spec.qspec), quat0, u_quat)

    # get time in the first axis, and return
    def trans(x):
        if len(x.shape) == 3:
            return jnp.transpose(x, axes=[1, 0, 2])
        else:
            return jnp.transpose(x)

    x_xyz = trans(x_xyz)
    y_xyz = trans(y_xyz)
    # x_yaw = trans(x_yaw)
    y_yaw = trans(y_yaw)
    # x_quat = trans(x_quat)
    # y_quat = trans(y_quat)

    return x_xyz, y_xyz, x_yaw, y_yaw, x_quat, y_quat


def head_dynamics(
    spec: mpc_spec.MPCSpec,
    xyz: jax.Array,
    yaw: jax.Array,
    quat: jax.Array,
    xyz_hist: jax.Array,
    yaw_hist: jax.Array,
    quat_hist: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute acceleration and angular velocity in the head frame.

    Parameters
    ----------
    spec :
        MPC specification.
    xyz :
        Linear translation, in table frame.
    yaw :
        Yaw angle, in table frame.
    tilt :
        Tilt vector, in table frame.
    xyz_hist :
        Past `xyz` history.
    yaw_hist :
        Past `yaw` history.
    quat_hist :
        Past `quat` history.

    Returns
    -------
    acc_head :
        Linear acceleration, in moving head frame.
    omega_head :
        Angular velocity, in moving head frame.
    """
    assert xyz_hist.shape == (2, 3)
    assert yaw_hist.shape == (2,)
    assert quat_hist.shape == (2, 4)
    xyz = jnp.vstack([xyz_hist, xyz])
    yaw = jnp.concatenate([yaw_hist, yaw])
    quat = jnp.concatenate([quat_hist, quat])

    def deriv(x):
        return jnp.diff(x, axis=0) / spec.dt

    def xyz_head_fun(delta, q):
        rot = comp.rot(q) - np.eye(3)
        return delta + rot @ spec.human_displacement

    def acc_head_fun(acc, q):
        rot = comp.rot(q)
        return rot.T @ (acc + mpc_spec.gravity)

    quat = jax.vmap(comp.inv_ty)(quat, yaw)
    xyz_head = jax.vmap(xyz_head_fun)(xyz, quat)
    acc_head = jax.vmap(acc_head_fun)(deriv(deriv(xyz_head)), quat[2:])
    omega_head = jax.vmap(comp.ang_vel, in_axes=[0, 0, None])(
        quat[1:-1], quat[2:], spec.dt
    )
    return acc_head, omega_head


def kinematics(
    spec: mpc_spec.MPCSpec,
    xyz: jax.Array,
    quat: jax.Array,
    yaw: jax.Array,
    xyz_hist: jax.Array,
    quat_hist: jax.Array,
    yaw_hist: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute kinematics for stewart platform.

    Parameters
    ----------
    spec :
        MPC specification.
    xyz :
        Linear translation, in table frame.
    quat :
        Unit quaternion, in table frame.
    yaw :
        Yaw angle, in table frame.
    xyz_hist :
        Past `xyz` history.
    quat_hist :
        Past `quat` history.
    yaw_hist :
        Past `yaw` history.

    Returns
    -------
    leg_pos :
        Stewart platform leg lengths.
    leg_vel :
        Stewart platform leg velocities.
    leg_pos_estop :
        Stewart platform leg lengths after pressing estop.
    leg_ang :
        Stewart platform joint angles (both top and bottom).
    euler_ang :
        Stewart platform euler angles.
    yaw_dot :
        Derivative of yaw euler angle.
    """
    assert xyz_hist.shape == (2, 3)
    assert quat_hist.shape == (2, 4)
    xyz = jnp.vstack([xyz_hist, xyz])
    quat = jnp.concatenate([quat_hist, quat])
    yaw = jnp.concatenate([yaw_hist, yaw])

    def table_geom(xyz, quat, yaw):
        rot = comp.rot(quat)
        rot_yaw = comp.rot_yaw(yaw)
        tops = spec.tops @ rot.T + xyz
        r_0_table = rot @ spec.r_0_table
        r_0_rotary = rot @ rot_yaw @ spec.r_0_rotary

        u = tops - spec.bots
        u_norm = jnp.linalg.norm(u, axis=1, keepdims=True)
        u /= u_norm
        leg_pos = jnp.squeeze(u_norm)

        vel_jac_inv = comp.inv6(comp.velocity_jacobian(spec.tops, u))
        a_f_table = comp.estop_a_f(
            -mpc_spec.gravity, spec.m_table, r_0_table, vel_jac_inv
        )
        a_f_rotary = comp.estop_a_f(
            -mpc_spec.gravity, spec.m_rotary, r_0_rotary, vel_jac_inv
        )
        a_f = a_f_table + a_f_rotary

        return tops, leg_pos, a_f

    tops, leg_pos, a_f = jax.vmap(table_geom)(xyz, quat, yaw)
    leg_vel = jnp.diff(leg_pos, axis=0) / spec.dt
    leg_ang = jax.vmap(comp.leg_ang, in_axes=[0, None, None, None])(
        tops, spec.bots, spec.top_normals, spec.bot_normals
    )
    euler_ang = jax.vmap(comp.quat2euler)(quat)
    yaw_dot = jnp.diff(yaw) / spec.dt

    # extra safety
    estop_delta_ell = functools.partial(
        comp.estop_delta_ell,
        spec.t_e,
        spec.a_b,
        spec.leg_safety_factor,
    )
    leg_pos_estop_delta = jax.vmap(estop_delta_ell)(leg_vel, a_f[1:])
    leg_pos_estop = leg_pos[1:] + leg_pos_estop_delta

    return (
        leg_pos[2:],
        leg_vel[1:],
        leg_ang[2:],
        leg_pos_estop[1:],
        euler_ang[2:],
        yaw_dot[1:],
    )


def eigen_vstates(
    spec: mpc_spec.MPCSpec,
    acc: jax.Array,
    omega: jax.Array,
    vstate0: jax.Array,
    return_eig_states: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Return vestibular states, using diagonalized state-space computations.

    Parameters
    ----------
    spec :
        MPC specification.
    acc_irl :
        Linear acceleration inputs.
    omega :
        Angular velociy inputs.
    vstate0 :
        Initial vestibular state by taking all vestibular internal states
        and flattening and concatenating them together.
    return_eig_states :
        If `True`, internal states for the vestibular system are eigen-states.
        If `False`, then internal states are converted back into their original
        basis representation.

    Returns
    -------
    x_lin :
        Internal states for linear dynamics of vestibular system.
    x_ang :
        Internal states for angular dynamics of vestibular system.
    y :
        Observed states for vestibular system.

    Warning
    -------
    The returned VState internal states should be interpreted as eigen-states.
    To get the correct internal states, you need to transform `P @ x` where
    the columns of `P` are the eigenvectors of `A`.
    This is functionality provided via the optional flag `return_eig_states`.
    """
    # setup
    s_ac = spec.vspec_acc
    s_jk = spec.vspec_jerk
    s_om = spec.vspec_omega

    s_ac_params = [s_ac.eig, s_ac.BP, s_ac.CP, s_ac.D, s_ac.P_inv, s_ac.P]
    s_jk_params = [s_jk.eig, s_jk.BP, s_jk.CP, s_jk.D, s_jk.P_inv, s_jk.P]
    s_om_params = [s_om.eig, s_om.BP, s_om.CP, s_om.D, s_om.P_inv, s_om.P]

    # partition
    a_num = 2 * s_ac.n_state
    j_num = s_jk.n_state
    # w_num = 3 * s_ome.n_state

    v0_a = vstate0[:a_num].reshape(2, -1)
    v0_j = vstate0[a_num : a_num + j_num]
    v0_w = vstate0[a_num + j_num :].reshape(3, -1)

    # integrate
    vmap_eigen_int = jax.vmap(
        functools.partial(
            siso.eigen_lti_int, return_eig_states=return_eig_states
        ),
        in_axes=[None] * 6 + [0, 1],
    )

    x_a, y_a = vmap_eigen_int(*s_ac_params, v0_a, acc[:, :2])
    x_j, y_j = siso.eigen_lti_int(
        *s_jk_params, v0_j, acc[:, 2], return_eig_states=return_eig_states
    )
    x_w, y_w = vmap_eigen_int(*s_om_params, v0_w, omega)

    # return
    # indices = (time, vals), with `x` vals being flattened and concat states
    x = jnp.hstack(
        [
            jnp.transpose(x_a, axes=[1, 0, 2]).reshape(x_a.shape[1], -1),
            x_j,
            jnp.transpose(x_w, axes=[1, 0, 2]).reshape(x_w.shape[1], -1),
        ]
    )
    y = jnp.hstack([y_a.T, y_j.reshape(-1, 1), y_w.T])
    return x, y
