r"""
We provide optimization components for Stewart platform MPC control.

* Tuning classes: :class:`Weights`, :class:`ExpWeights`, and :class:`CostTerms`.
* The cost function implementation: :func:`cost_flat_jax`.
* A feedback loop for python simulations: :func:`train_step_with_cost`.

The general philosophy is as follows.

* Functions are jax compatible.
* Implementations are hacky (for easy experimentation).
* The number of abstractions should be minimized.

See the :doc:`C++ docs <../cpp>` to see how to integrate the MPC feedback
into a C++ program.

MPC Formulation
===============

Our optimal control problem takes the usual form

.. math::

    u^*(\cdot) = \operatorname*{arg\,min}_{u \in L^2([0, T])} \int_0^T
    \ell(t, x(t), u(t)) \operatorname*{d}\!t + K(x(T))

subject to some linear dynamics

.. math::

    \dot{x} = A x + B u, \quad x(0) = x_0.

The controls :math:`u` are the linear accelerations and euler-angle
accelerations, in the head frame.
Our state vector :math:`x` can be decomposed as

.. math::

    x = \begin{bmatrix} x_{\mathrm{robo}} \\ x_{\mathrm{vest}} \end{bmatrix},
    \quad x_{\mathrm{vest}} = \begin{bmatrix} x_{\mathrm{irl}} \\
    x_{\mathrm{sim}} \end{bmatrix}.

where :math:`x_{\mathrm{robo}}` denotes the Cartesian position and velocity of
the headframe and where :math:`x_{\mathrm{irl}}` and :math:`x_{\mathrm{sim}}`
denote the vestibular linear acceleration and angular velocity of the real
person and simulated person, respectively.
The dynamics :math:`(A, B)` act on :math:`x_{\mathrm{robo}}` as a double
integrator, and they act on :math:`x_{\mathrm{irl}}` and
:math:`x_{\mathrm{sim}}` via SISO vestibular models, taken from the literature.
The SISO vestibular systems are described by
:class:`exp_mpc.stewart_min.vest.VSpec`.
Next, the running cost is of the form

.. math::

    \ell(t, x, u) \approx |(x_{\mathrm{irl}} - x_{\mathrm{sim}}) \odot
    e^{\alpha t}|_W^2 + \sum_{c \in \mathcal{C}} q_c(x_{\mathrm{robo}}) +
    |u|_W^2.

Namely, we have exponential decay factors on the vestibular tracking terms, and
we have a sum over the boundary quartic costs :math:`q_c`, where
:math:`\mathcal{C}` denotes the set of constraints.
The quartic costs are implemented as
:class:`exp_mpc.stewart_min.quartic_cost.QuarticCost`.
Finally, we solve the minimization problem via real-time L-BFGS iterations.
The L-BFGS algorithm is implemented in a separate library.
"""

from __future__ import annotations

import dataclasses
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as sci_opt
from lbfgs import lbfgs

from exp_mpc.stewart_min import comp, mpc_spec, siso, utils

# lbfgs_res = (minimizer, value, gradient)
type LBFGSResult = tuple[jax.Array, jax.Array, jax.Array]

# make sure to enable 64-bit precision for jax
# this is necessary for good performance
# use the following line when importing this library
jax.config.update("jax_enable_x64", True)


########
# cost #
########


def _head_cost(
    spec: mpc_spec.MPCSpec,
    y_vest_irl: jax.Array,
    y_vest_sim: jax.Array,
) -> jax.Array:
    n = y_vest_irl.shape[0]

    lin_diff = jnp.square(y_vest_irl[:, :3] - y_vest_sim[:, :3])
    lin_cost = spec.weights.scale_lin_dyn(n) * lin_diff
    ome_diff = jnp.square(y_vest_irl[:, 3:] - y_vest_sim[:, 3:])
    ome_cost = spec.weights.scale_omega(n) * ome_diff

    return 0.5 * jnp.sum(jnp.mean(lin_cost + ome_cost, axis=0))


def _ik_cost(
    spec: mpc_spec.MPCSpec,
    leg_pos: jax.Array,
    leg_vel: jax.Array,
    leg_ang: jax.Array,
    leg_pos_estop: jax.Array,
) -> jax.Array:
    n = leg_pos.shape[0]

    leg_pos_fun = jax.vmap(spec.cost_terms.leg_pos_cost)
    leg_vel_fun = jax.vmap(spec.cost_terms.leg_vel_cost)
    leg_ang_fun = jax.vmap(spec.cost_terms.leg_ang_cost)

    leg_pos_quart = leg_pos_fun(leg_pos.flatten()).reshape(-1, 6)
    leg_vel_quart = leg_vel_fun(leg_vel.flatten()).reshape(-1, 6)
    leg_ang_quart = leg_ang_fun(leg_ang.flatten()).reshape(-1, 12)
    leg_pos_estop_quart = leg_pos_fun(leg_pos_estop.flatten()).reshape(-1, 6)

    def mean(x):
        return jnp.sum(jnp.mean(x, axis=0))

    leg_pos_cost = mean(leg_pos_quart * spec.weights.scale_leg_pos(n))
    leg_vel_cost = mean(leg_vel_quart * spec.weights.scale_leg_vel(n))
    leg_ang_cost = mean(leg_ang_quart * spec.weights.scale_leg_ang(n))
    leg_pos_estop_cost = mean(
        leg_pos_estop_quart * spec.weights.scale_leg_pos(n)
    )
    return leg_pos_cost + leg_vel_cost + leg_ang_cost + leg_pos_estop_cost


def _euler_cost(
    spec: mpc_spec.MPCSpec,
    euler_ang: jax.Array,
    yaw: jax.Array,
    yaw_dot: jax.Array,
    yaw_ctrl: jax.Array,
) -> jax.Array:
    n = euler_ang.shape[0]

    roll_fun = jax.vmap(spec.cost_terms.roll_cost)
    pitch_fun = jax.vmap(spec.cost_terms.pitch_cost)
    yaw_fun = jax.vmap(spec.cost_terms.yaw_cost)
    yaw_dot_fun = jax.vmap(spec.cost_terms.yaw_dot_cost)
    yaw_ctrl_fun = jax.vmap(spec.cost_terms.yaw_ctrl_cost)

    roll_cost = roll_fun(euler_ang[:, 0]) * spec.weights.scale_roll(n)
    pitch_cost = pitch_fun(euler_ang[:, 1]) * spec.weights.scale_pitch(n)
    yaw_cost = yaw_fun(yaw) * spec.weights.scale_yaw(n)
    yaw_dot_cost = yaw_dot_fun(yaw_dot) * spec.weights.scale_yaw_dot(n)
    yaw_ctrl_cost = yaw_ctrl_fun(yaw_ctrl) * spec.weights.scale_yaw_ctrl(n)

    return jnp.mean(
        roll_cost + pitch_cost + yaw_cost + yaw_dot_cost + yaw_ctrl_cost
    )


def _control_cost(
    spec: mpc_spec.MPCSpec,
    control: mpc_spec.MPCSpec,
) -> jax.Array:
    control = control.reshape(-1, 6)
    n = control.shape[0]
    control_cost = jnp.square(control) * spec.weights.scale_control(n)
    return 0.5 * jnp.sum(jnp.mean(control_cost, axis=0))


def _terminal_cost(
    spec: mpc_spec.MPCSpec,
    terminal_param: jax.Array,
    xyz: jax.Array,
    yaw: jax.Array,
    quat: jax.Array,
) -> jax.Array:
    if not spec.use_terminal:
        return 0.0

    def _ssum(x):
        return jnp.sum(jnp.square(x))

    scale = jnp.exp(-spec.weights.terminal_exp_scale * terminal_param)

    cart_last = jnp.concatenate([xyz[-1], jnp.atleast_1d(yaw[-1]), quat[-1]])
    ang_home = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])
    cart_home = jnp.concatenate([spec.cart_home, ang_home])

    rt_cost = _ssum(cart_last - cart_home)
    res = scale * rt_cost * spec.weights.terminal_rt_scale
    return res


def cost(
    spec: mpc_spec.MPCSpec,  # static
    control: jax.Array,
    prefilt0: jax.Array,
    filt0: jax.Array,
    vstate0_irl: jax.Array,
    y_vest_sim: jax.Array,
    terminal_param: jax.Array,
    xyz_hist: jax.Array,
    yaw_hist: jax.Array,
    quat_hist: jax.Array,
) -> jax.Array:
    """Evaluate MPC objective from a control trajectory.

    Parameters
    ----------
    spec :
        MPC specification.
    control :
        Flattened control sequence with ordering
        `[x, y, z, yaw, tilt0, tilt1]` per time step.
    prefilt0 :
        Initial states for prefilter.
    filt0 :
        Initial states for control filters.
    vstate0_irl :
        Initial vestibular state for the in-real-life person.
    y_vest_sim :
        Observed vestibular states for simulated person.
    terminal_param :
        Parameter for terminal cost.
        For now, it denotes when the robot should return to home.
    xyz_hist :
        Previous two linear positions of robot.
    yaw_hist :
        Previous two yaw positions of robot.
    quat_hist :
        Previous two unit quaternions of robot.

    Returns
    -------
    cost :
        Scalar MPC objective value.
    """
    # compute states
    _, y_pre = utils.prefilt_u(
        spec=spec,
        u=control,
        prefilt0=prefilt0,
    )
    u_yaw = y_pre[:, 3]
    _, y_xyz, _, y_yaw, _, y_quat = utils.apply_u(
        spec=spec,
        u=y_pre,
        filt0=filt0,
    )
    acc_head, omega_head = utils.head_dynamics(
        spec=spec,
        xyz=y_xyz,
        yaw=y_yaw,
        quat=y_quat,
        xyz_hist=xyz_hist,
        yaw_hist=yaw_hist,
        quat_hist=quat_hist,
    )
    leg_pos, leg_vel, leg_ang, leg_pos_estop, euler_ang, yaw_dot = (
        utils.kinematics(
            spec=spec,
            xyz=y_xyz,
            quat=y_quat,
            yaw=y_yaw,
            xyz_hist=xyz_hist,
            quat_hist=quat_hist,
            yaw_hist=yaw_hist,
        )
    )
    _, y_vest_irl = utils.eigen_vstates(
        spec=spec,
        acc=acc_head,
        omega=omega_head,
        vstate0=vstate0_irl,
        return_eig_states=True,
    )

    # cost
    cost_head = _head_cost(spec, y_vest_irl, y_vest_sim)
    cost_ik = _ik_cost(spec, leg_pos, leg_vel, leg_ang, leg_pos_estop)
    cost_euler = _euler_cost(spec, euler_ang, y_yaw, yaw_dot, u_yaw)
    cost_control = _control_cost(spec, control)
    cost_term = _terminal_cost(spec, terminal_param, y_xyz, y_yaw, y_quat)
    return cost_head + cost_ik + cost_euler + cost_control + cost_term


################
# jax training #
################


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TrainState:
    """State for training, plus extra info for post processing.

    The name was motivated by the machine learning community.
    See :class:`flax.training.train_state.TrainState`.
    Essentially, the container includes the updated parameters after each MPC
    optimization iteration.

    Parameters
    ----------
    control :
        Control sequence for the MPC horizon.
        (Last optimization solution.)
    prefilt0 :
        Initial state for control pre-filtering.
        Includes initial unit quaternion (for pre-filting control).
    filt0 :
        Initial state for robot filters.
    vstate0_irl :
        Current vestibular state for the in-real-life person.
    vstate0_sim :
        Current vestibular state for the simulated/reference person.
    y_vest_sim_hist :
        Previous vestibular states for the simulated/reference person.
    xyz_hist :
        Previous two linear positions of robot.
    yaw_hist :
        Previous two yaw positions of robot.
    quat_hist :
        Previous two unit quaternions of robot.
    terminal_param :
        Parameter for terminal cost.
    iter :
        Train state identifier.
        (Number of iterations through the MPC algorithm.)
    x_pre :
        Internal states for control prefilter.
    y_pre :
        Controls for robots.
    x_xyz :
        Internal states for robot linear position.
    y_xyz :
        Observed states for robot linear position.
    x_yaw :
        Internal states for robot yaw angle.
    u_yaw :
        Acutal (nondiff) controls for robot yaw angle.
    y_yaw :
        Observed states for robot yaw angle.
    x_quat :
        Internal states for robot quaternion.
    y_quat :
        Observed states for robot quaternion.
    acc_head :
        Linear acceleration of head frame.
    omega_head :
        Angular velocity of head frame.
    leg_pos :
        Leg positions from kinematics.
    leg_vel :
        Leg velocities from kinematics.
    leg_ang :
        Leg angles from kinematics.
    leg_pos_estop :
        Leg position after estop.
    euler_ang :
        Euler angles from kinematics.
    yaw_dot :
        Derivative of yaw angle from kinematics.
    x_vest_irl :
        Internal states for vestibular system of in-real-life person.
    y_vest_irl :
        Observed states for vestibular system of in-real-life person.
    x_vest_sim :
        Internal states for vestibular system of simulated person.
    y_vest_sim :
        Observed states for vestibular system of simulated person.
    """

    # training info
    control: jax.Array
    prefilt0: jax.Array
    filt0: jax.Array
    vstate0_irl: jax.Array
    vstate0_sim: jax.Array
    y_vest_sim_hist: jax.Array
    xyz_hist: jax.Array
    yaw_hist: jax.Array
    quat_hist: jax.Array
    terminal_param: jax.Array
    iter: jax.Array

    # extra info
    x_pre: jax.Array
    y_pre: jax.Array
    x_xyz: jax.Array
    y_xyz: jax.Array
    x_yaw: jax.Array
    u_yaw: jax.Array
    y_yaw: jax.Array
    x_quat: jax.Array
    y_quat: jax.Array
    acc_head: jax.Array
    omega_head: jax.Array
    leg_pos: jax.Array
    leg_vel: jax.Array
    leg_ang: jax.Array
    leg_pos_estop: jax.Array
    euler_ang: jax.Array
    yaw_dot: jax.Array
    x_vest_irl: jax.Array
    y_vest_irl: jax.Array
    x_vest_sim: jax.Array
    y_vest_sim: jax.Array

    @classmethod
    def zero_init(
        cls,
        spec: mpc_spec.MPCSpec,
        sim_acc_z: float = 9.81,
    ) -> TrainState:
        """Init train state with zeros.

        Parameters
        ----------
        spec :
            MPC specification.
        sim_acc_z :
            Initial simulation acceleration in the z-direction.
            E.g., 1.625 for moon gravity and 9.81 for earth gravity.

        Returns
        -------
        train_state :
            Zeroed train state.
        """
        acc_num = spec.vspec_acc.n_state
        jerk_num = spec.vspec_jerk.n_state
        omega_num = spec.vspec_omega.n_state
        u_num = 6
        v_num = 2 * acc_num + jerk_num + 3 * omega_num

        control = jnp.zeros(u_num * spec.n)

        # prefilt with identity tilt
        tilt0 = jnp.array([1.0, 0.0, 0.0])
        ctrl_home = np.concatenate([spec.cart_home, np.zeros(3)])
        prefilt0_terms = []
        for pos in ctrl_home:
            prefilt0_pos = siso.obs_x0(
                A=spec.ctrlspec.A,
                B=spec.ctrlspec.B,
                C=spec.ctrlspec.C,
                D=spec.ctrlspec.D,
                y=np.ones(spec.ctrlspec.n_state) * pos,
                u=np.ones(spec.ctrlspec.n_state) * pos,
            )
            prefilt0_terms.append(prefilt0_pos)
        prefilt0 = jnp.concatenate(prefilt0_terms + [tilt0])

        # produce filt0 with home at xyz home and identity rotation
        filt0_terms = []
        home = np.concatenate(
            [spec.cart_home, np.array([0.0, 1.0, 0.0, 0.0, 0.0])]
        )
        filt_specs = [spec.xyzspec] * 3 + [spec.yspec] + [spec.qspec] * 4
        for pos, filt_spec in zip(home, filt_specs):  # xyz
            filt0_pos = siso.obs_x0(
                A=filt_spec.A,
                B=filt_spec.B,
                C=filt_spec.C,
                D=filt_spec.D,
                y=np.ones(filt_spec.n_state) * pos,
                u=np.ones(filt_spec.n_state) * pos,
            )
            filt0_terms.append(filt0_pos)
        filt0 = jnp.concatenate(filt0_terms)

        # need to initialize z-jerk carefully
        vstate0_irl = jnp.zeros(v_num)
        vstate0_sim = jnp.zeros(v_num)

        def jerk0(val):
            return siso.obs_x0(
                A=spec.vspec_jerk.A,
                B=spec.vspec_jerk.B,
                C=spec.vspec_jerk.C,
                D=spec.vspec_jerk.D,
                y=np.zeros(spec.vspec_jerk.n_state),
                u=np.ones(spec.vspec_jerk.n_state) * val,
            )

        jerk0_earth = jerk0(mpc_spec.gravity[-1])
        jerk0_sim = jerk0(sim_acc_z)
        n_acc = spec.vspec_acc.n_state
        n_jerk = spec.vspec_jerk.n_state
        idxs = slice(2 * n_acc, 2 * n_acc + n_jerk)
        vstate0_irl = vstate0_irl.at[idxs].set(jerk0_earth)
        vstate0_sim = vstate0_sim.at[idxs].set(jerk0_sim)

        # misc
        y_vest_sim_hist = jnp.zeros((4, 6))
        xyz_hist = jnp.tile(spec.cart_home.reshape(1, -1), reps=(2, 1))
        yaw_hist = jnp.zeros((2,))
        quat_hist = jnp.tile(
            jnp.array([1.0, 0.0, 0.0, 0.0]).reshape(1, -1), reps=(2, 1)
        )
        terminal_param = jnp.array(0.0, dtype=jnp.float64)
        iter = jnp.array(0, dtype=jnp.int64)

        x_pre = jnp.zeros((spec.n, spec.ctrlspec.n_state * 6))
        y_pre = jnp.zeros((spec.n, 7))
        x_xyz = jnp.zeros((spec.n, 3, spec.xyzspec.n_state))
        y_xyz = jnp.zeros((spec.n, 3))
        x_yaw = jnp.zeros((spec.n, spec.yspec.n_state))
        u_yaw = jnp.zeros((spec.n,))
        y_yaw = jnp.zeros((spec.n,))
        x_quat = jnp.zeros((spec.n, 4, spec.qspec.n_state))
        y_quat = jnp.transpose(
            jnp.vstack([jnp.ones(spec.n)] + [jnp.zeros(spec.n)] * 3)
        )  # identity
        acc_head = jnp.zeros((spec.n, 3))
        omega_head = jnp.zeros((spec.n, 3))
        leg_pos = jnp.ones((spec.n, 6)) * mpc_spec._lengths_home
        leg_vel = jnp.zeros((spec.n, 6))
        leg_ang = jnp.zeros((spec.n, 12))
        leg_pos_estop = jnp.ones((spec.n, 6)) * (
            mpc_spec._lengths_home - 0.03589903
        )
        euler_ang = jnp.zeros((spec.n, 3))
        yaw_dot = jnp.zeros((spec.n,))
        x_vest_irl = jnp.zeros((spec.n, v_num))
        y_vest_irl = jnp.zeros((spec.n, 6))
        x_vest_sim = jnp.zeros((spec.n, v_num))
        y_vest_sim = jnp.zeros((spec.n, 6))

        return cls(
            control=control,
            prefilt0=prefilt0,
            filt0=filt0,
            vstate0_irl=vstate0_irl,
            vstate0_sim=vstate0_sim,
            y_vest_sim_hist=y_vest_sim_hist,
            xyz_hist=xyz_hist,
            yaw_hist=yaw_hist,
            quat_hist=quat_hist,
            terminal_param=terminal_param,
            iter=iter,
            x_pre=x_pre,
            y_pre=y_pre,
            x_xyz=x_xyz,
            y_xyz=y_xyz,
            x_yaw=x_yaw,
            u_yaw=u_yaw,
            y_yaw=y_yaw,
            x_quat=x_quat,
            y_quat=y_quat,
            acc_head=acc_head,
            omega_head=omega_head,
            leg_pos=leg_pos,
            leg_vel=leg_vel,
            leg_ang=leg_ang,
            leg_pos_estop=leg_pos_estop,
            euler_ang=euler_ang,
            yaw_dot=yaw_dot,
            x_vest_irl=x_vest_irl,
            y_vest_irl=y_vest_irl,
            x_vest_sim=x_vest_sim,
            y_vest_sim=y_vest_sim,
        )


def lbfgs_cost(
    spec: mpc_spec.MPCSpec,
    train_state: TrainState,
    y_vest_sim: jax.Array,
    terminal_param: jax.Array,
    args: None,
    control: jax.Array,
) -> jax.Array:
    """L-BFGS wrapper of :func:`cost`.

    Parameters
    ----------
    spec :
        MPC specification.
    args :
        Tuple ``(train_state, acc_ref, omega_ref)`` passed through L-BFGS.
        These are the arguments that change during each MPC control cycle.
    control :
        Control sequence being optimized.

    Returns
    -------
    cost :
        Scalar MPC objective value.
    """
    return cost(
        spec=spec,
        control=control,
        prefilt0=train_state.prefilt0,
        filt0=train_state.filt0,
        vstate0_irl=train_state.vstate0_irl,
        y_vest_sim=y_vest_sim,
        terminal_param=terminal_param,
        xyz_hist=train_state.xyz_hist,
        yaw_hist=train_state.yaw_hist,
        quat_hist=train_state.quat_hist,
    )


lbfgs_cost_and_grad = jax.jit(jax.value_and_grad(lbfgs_cost, argnums=-1))

prefilt_u = jax.jit(utils.prefilt_u)
apply_u = jax.jit(utils.apply_u)
head_dynamics = jax.jit(utils.head_dynamics)
kinematics = jax.jit(utils.kinematics)
eigen_vstates = jax.jit(
    utils.eigen_vstates,
    static_argnames=["return_eig_states"],
)


def apply_control(
    spec: mpc_spec.MPCSpec,
    train_state: TrainState,
    control: jax.Array,
    x_vest_sim: jax.Array,
    y_vest_sim: jax.Array,
    terminal_param: jax.Array,
) -> TrainState:
    """Apply control and references for new TrainState.

    Parameters
    ----------
    spec :
        MPC specification.
    train_state :
        Current MPC state.
    control :
        New control to apply.
    x_vest_sim :
        Internal states for simulated vestibular system.
    y_vest_sim :
        Observed states for simulated vestibular system.
    terminal_param :
        Parameter for terminal cost.

    Returns
    -------
    next_state :
        Updated MPC state.
    """
    ts = train_state

    # compute states
    # (code is mostly duplicated from `cost`)
    x_pre, y_pre = prefilt_u(
        spec=spec,
        u=control,
        prefilt0=ts.prefilt0,
    )
    u_yaw = y_pre[:, 3]
    x_xyz, y_xyz, x_yaw, y_yaw, x_quat, y_quat = apply_u(
        spec=spec,
        u=y_pre,
        filt0=ts.filt0,
    )
    acc_head, omega_head = head_dynamics(
        spec=spec,
        xyz=y_xyz,
        yaw=y_yaw,
        quat=y_quat,
        xyz_hist=ts.xyz_hist,
        yaw_hist=ts.yaw_hist,
        quat_hist=ts.quat_hist,
    )
    leg_pos, leg_vel, leg_ang, leg_pos_estop, euler_ang, yaw_dot = kinematics(
        spec=spec,
        xyz=y_xyz,
        quat=y_quat,
        yaw=y_yaw,
        xyz_hist=ts.xyz_hist,
        quat_hist=ts.quat_hist,
        yaw_hist=ts.yaw_hist,
    )
    x_vest_irl, y_vest_irl = eigen_vstates(
        spec=spec,
        acc=acc_head,
        omega=omega_head,
        vstate0=ts.vstate0_irl,
        return_eig_states=False,
    )

    # bookkeeping
    prefilt0 = jnp.concatenate([x_pre[0], y_pre[0][-3:]])
    filt0 = jnp.vstack(
        [x_xyz[0], x_yaw[0].reshape(1, *x_yaw[0].shape), x_quat[0]]
    )
    y_vest_sim_hist = jnp.vstack(
        [ts.y_vest_sim_hist[1:], jnp.atleast_2d(y_vest_sim[0])]
    )
    next_state = TrainState(
        control=control,
        prefilt0=prefilt0,
        filt0=jnp.ravel(filt0),
        vstate0_irl=x_vest_irl[0],
        vstate0_sim=x_vest_sim[0],
        y_vest_sim_hist=y_vest_sim_hist,
        xyz_hist=jnp.vstack([ts.xyz_hist[-1], y_xyz[0]]),
        yaw_hist=jnp.array([ts.yaw_hist[-1], y_yaw[0]]),
        quat_hist=jnp.vstack([ts.quat_hist[-1], y_quat[0]]),
        terminal_param=terminal_param,
        iter=ts.iter + 1,
        x_pre=x_pre,
        y_pre=y_pre,
        x_xyz=x_xyz,
        y_xyz=y_xyz,
        x_yaw=x_yaw,
        u_yaw=u_yaw,
        y_yaw=y_yaw,
        x_quat=x_quat,
        y_quat=y_quat,
        acc_head=acc_head,
        omega_head=omega_head,
        leg_pos=leg_pos,
        leg_vel=leg_vel,
        leg_ang=leg_ang,
        leg_pos_estop=leg_pos_estop,
        euler_ang=euler_ang,
        yaw_dot=yaw_dot,
        x_vest_irl=x_vest_irl,
        y_vest_irl=y_vest_irl,
        x_vest_sim=x_vest_sim,
        y_vest_sim=y_vest_sim,
    )
    return next_state


def predict_vestibular(
    spec: mpc_spec.MPCSpec,
    train_state: TrainState,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Predict vestibular states for prediction horizon.

    There are two possible outputs.
    If `acc_ref` and `omega_ref` are single inputs, i.e., only vectors of
    length 3, then we predict the future vestibular horizon.
    If `acc_ref` and `omega_ref` are given across the entire horizon, i.e.,
    they each have shape `(n, 3)` where `n` is the prediction horizon length,
    then the vestibular model is directl integrated using these reference
    values.

    Parameters
    ----------
    spec :
        MPC specification.
    train_state :
        Current MPC state.
    acc_ref :
        Reference linear acceleration trajectory.
    omega_ref :
        Reference angular velocity trajectory.

    Returns
    -------
    x_vest_sim :
        Internal states for vestibular model.
    y_vest_sim :
        Observed (predicted) states for vestiubular model.
    """
    # setup
    assert acc_ref.shape[-1] == 3
    one_data = len(acc_ref.shape) == 1
    if one_data:
        acc_ref = jnp.tile(acc_ref.reshape(1, -1), reps=(spec.n, 1))
        omega_ref = jnp.tile(omega_ref.reshape(1, -1), reps=(spec.n, 1))
    assert acc_ref.shape == omega_ref.shape
    assert acc_ref.shape[0] == spec.n
    x_vest_sim, y_vest_sim = eigen_vstates(
        spec=spec,
        acc=acc_ref,
        omega=omega_ref,
        vstate0=train_state.vstate0_sim,
        return_eig_states=False,
    )

    # predict into the future?
    if one_data:
        y_vest_sim_hist = jnp.vstack(
            [
                train_state.y_vest_sim_hist[1:],
                jnp.atleast_2d(y_vest_sim[0]),
            ]
        )

        def running_pred():
            pred_hist = jax.vmap(
                comp.pred_hist,
                in_axes=[0, None, None, 0, 1],
            )
            y_vest_sim = pred_hist(
                spec.pred_n, spec.n, spec.dt, spec.pred_E, y_vest_sim_hist
            )
            return jnp.transpose(y_vest_sim)

        def initial_pred():
            # too early?
            return y_vest_sim

        # the 50 constant can reasonably be made 3...
        y_vest_sim = jax.lax.cond(
            train_state.iter > 50, running_pred, initial_pred
        )
    return x_vest_sim, y_vest_sim


def train_step_with_cost_jax(
    spec: mpc_spec.MPCSpec,
    train_state: TrainState,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    opt_scheme: str = "jax",
) -> tuple[TrainState, LBFGSResult]:
    """Run one MPC control cycle with JAX L-BFGS.

    Parameters
    ----------
    spec :
        MPC specification.
    train_state :
        Current MPC state.
    acc_ref :
        Reference linear acceleration trajectory.
    omega_ref :
        Reference angular velocity trajectory.
    opt_scheme :
        Determine which optimizer to use.
        Valid options include `["jax", "scipy", "none"]`.

    Returns
    -------
    next_state :
        Updated MPC state.
    lbfgs_res :
        L-BFGS optimizer tuple `(minimizer, value, gradient)`.
    """
    ts = train_state

    # update intial guess of solution
    guess = ts.control.reshape(-1, 6)[1:]  # skip initial
    guess_last = guess[-1]
    guess = jnp.vstack([guess, guess_last.reshape(1, -1)])
    guess_flat = jnp.ravel(guess)

    # vestibular_prediction
    x_vest_sim, y_vest_sim = predict_vestibular(spec, ts, acc_ref, omega_ref)

    # terminal param
    tp0 = train_state.terminal_param
    tp1 = jnp.sum(jnp.square(acc_ref)) + jnp.sum(jnp.square(omega_ref))
    alpha = spec.alpha_terminal
    terminal_param = (1 - alpha) * tp0 + alpha * tp1
    # terminal_param = tp1

    # compute
    opt_fun = functools.partial(
        lbfgs_cost_and_grad, spec, train_state, y_vest_sim, terminal_param
    )
    if opt_scheme.lower() == "jax":
        opt_params = lbfgs.OptParamsLBFGS(
            fun=opt_fun,
            max_iter=spec.max_iter,
            max_ls=spec.max_ls,
            init_norm=spec.init_norm,
            debug=spec.debug,
            unroll=spec.unroll,
        )
        res = lbfgs.lbfgs(
            opt_params=opt_params,
            x0=guess_flat,
            fun_params=None,
        )
        opt_control = res[0]
    elif opt_scheme.lower() == "scipy":
        res_sci = sci_opt.minimize(
            fun=functools.partial(opt_fun, None),  # lbfgs library shenanigans
            x0=guess_flat,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": spec.max_iter,
                "maxls": spec.max_ls,
            },
        )
        res = (res_sci.x, res_sci.fun, res_sci.jac)
        opt_control = res[0]
    else:
        opt_control = ts.control  # just use the given control
        res = None

    next_state = apply_control(
        spec, train_state, opt_control, x_vest_sim, y_vest_sim, terminal_param
    )
    return next_state, res


train_step_with_cost_jit = jax.jit(
    train_step_with_cost_jax,
    static_argnames=["opt_scheme"],
)


def train_step_with_cost(
    spec: mpc_spec.MPCSpec,
    train_state: TrainState,
    acc_ref: jax.Array,
    omega_ref: jax.Array,
    opt_scheme: str = "jax",
) -> tuple[TrainState, LBFGSResult, float]:
    """Run one MPC control cycle with JAX L-BFGS, and measure wall time.

    Parameters
    ----------
    acc_ref :
        Reference linear acceleration trajectory.
    omega_ref :
        Reference angular velocity trajectory.
    train_state :
        Current MPC state.
    spec :
        MPC specification.
    opt_scheme :
        Determine which optimizer to use.
        Valid options include `["jax", "scipy", "none"]`.

    Returns
    -------
    next_state :
        Updated MPC state.
    lbfgs_res :
        L-BFGS optimizer tuple `(minimizer, value, gradient)`.
    elapsed_time :
        Wall-time in seconds for calling the jit-ed
        :func:`train_step_with_cost_jax`.
    """
    t0 = time.time()
    if opt_scheme.lower() in ["jax", "none"]:
        train_step = train_step_with_cost_jit
    else:
        train_step = train_step_with_cost_jax
    res = train_step(
        acc_ref=acc_ref,
        omega_ref=omega_ref,
        train_state=train_state,
        spec=spec,
        opt_scheme=opt_scheme,
    )
    res[0].filt0.block_until_ready()  # wait for computation for good timing
    t1 = time.time()
    return res[0], res[1], t1 - t0
