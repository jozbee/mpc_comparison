"""
Dataclasses that specify relevant Stewart platform parameters and geometry.
Many defaults are defined here for testing.
"""

from __future__ import annotations
import dataclasses
import typing as tp
import numpy as np
import control as ct
import jax
import jax.numpy as jnp

import exp_mpc.stewart_min.siso as siso
import exp_mpc.stewart_min.quartic_cost as quartic_cost

jax.config.update("jax_enable_x64", True)


# dataclass helpers
def _default_elem(elem):
    return dataclasses.field(default_factory=lambda: elem)


def _init_field(arr: jax.Array) -> jax.Array:
    """Initialize a field with the same shape as the input array."""
    return dataclasses.field(
        default_factory=lambda: arr,
    )


# constants useful in the same context as robots
# (maybe should be specified somewhere else, but a special `const.py` file for
#  them seems a bit much)
gravity = np.array([0.0, 0.0, 9.81])
moon_gravity = np.array([0.0, 0.0, 1.625])

# geometry constants
_human_displacement = np.array([-0.3302, 0.0, 1.2977])
_bots = np.array(
    [
        [952.5055, 91.0723, -1410.0000],
        [-398.5396, 869.5826, -1409.8621],
        [-555.4801, 779.1038, -1410.0000],
        [-555.0219, -779.3507, -1409.6010],
        [-398.5396, -869.9006, -1410.0000],
        [952.7381, -89.7865, -1409.8718],
    ]
)
_bots *= 1e-3
_tops = np.array(
    [
        [314.3190, 327.5610, -215.5590],
        [126.7200, 435.8100, -215.4460],
        [-441.1510, 107.1130, -215.1930],
        [-441.5680, -109.0240, -215.6620],
        [126.7200, -436.9520, -215.6520],
        [314.7630, -328.8960, -215.7820],
    ]
)
_tops *= 1e-3

_bot_normals = np.array(
    [
        [-0.435014, 0.162005, 0.885729],
        [0.357803, -0.295803, 0.885708],
        [0.077200, -0.457799, 0.885698],
        [0.077200, 0.457799, 0.885698],
        [0.357803, 0.295803, 0.885708],
        [-0.435014, -0.162005, 0.885729],
    ]
)
_bot_normals /= np.linalg.norm(_bot_normals, axis=1)[:, np.newaxis]
_top_normals = np.array(
    [
        [0.435014, -0.162005, -0.885729],
        [-0.357803, 0.295803, -0.885708],
        [-0.077200, 0.457799, -0.885698],
        [-0.077200, -0.457799, -0.885698],
        [-0.357803, -0.295803, -0.885708],
        [0.435014, 0.162005, -0.885729],
    ]
)
_top_normals /= np.linalg.norm(_top_normals, axis=1)[:, np.newaxis]

_cart_home = np.array([0.0, 0.0, 0.1])  # home cartesian translation
_tops_home = np.array([top + _cart_home for top in _tops])
_lengths_home = float(np.mean(np.linalg.norm(_tops_home - _bots, axis=1)))

# vspec ref defs
# dt = 0.005
dt = 0.01

_s = ct.tf("s")
transfer_acc0 = 0.911 * (_s + 0.0988)
transfer_acc0 /= (_s + 0.133) * (_s + 1.95)
vspec_acc0 = siso.DiscreteEigSISO.cont2discrete(transfer_acc0, dt)

transfer_jerk0 = _s * transfer_acc0
vspec_jerk0 = siso.DiscreteEigSISO.cont2discrete(transfer_jerk0, dt)

transfer_omega0 = 10.3 * _s * 30 * _s
transfer_omega0 /= (10.2 * _s + 1) * (0.1 * _s + 1) * (30 * _s + 1)
vspec_omega0 = siso.DiscreteEigSISO.cont2discrete(transfer_omega0, dt)

transfer_omega1 = 5.73 * 80 * _s**2 * (1 + 0.06 * _s)
transfer_omega1 /= (1 + 80 * _s) * (1 + 5.73 * _s) * (1 + 0.005 * _s)
vspec_omega1 = siso.DiscreteEigSISO.cont2discrete(transfer_omega1, dt)

transfer_omega2 = 5.73 * 80 * _s**2
transfer_omega2 /= (1 + 80 * _s) * (1 + 5.73 * _s)
vspec_oemga2 = siso.DiscreteEigSISO.cont2discrete(transfer_omega2, dt)

spec_refs: dict[str, tuple[ct.TransferFunction, siso.DiscreteEigSISO]] = {
    "acc0": (transfer_acc0, vspec_acc0),
    "jerk0": (transfer_jerk0, vspec_jerk0),
    "omega0": (transfer_omega0, vspec_omega0),
    "omega1": (transfer_omega1, vspec_omega1),
    "omega2": (transfer_omega2, vspec_oemga2),
}

# pre-filtering of controls ref def
transfer_cspec = 1 / (_s * 1 / (2 * np.pi) + 1) ** 3
cspec = siso.DiscreteSISO.cont2discrete(transfer_cspec, dt, "zoh")

# control filtering ref defs
# (pose spec and rotary spec)
transfer_pspec = 1 / (_s * 1 / (2 * np.pi * 2) + 1)
pspec = siso.DiscreteSISO.cont2discrete(transfer_pspec, dt, "bilinear")
transfer_respec = 1 / (_s * 1 / (2 * np.pi * 1) + 1)
rspec = siso.DiscreteSISO.cont2discrete(transfer_respec, dt, "bilinear")


###############################
# weight and cost bookkeeping #
###############################


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Weights:
    """Cost function weights.

    Parameters
    ----------
    lin_dyn :
        Per-axis weights for linear dynamics: ``[x_dot2, y_dot2, z_dot3]``.
    omega :
        Per-axis weights for angular velocity:
        ``[roll_dot, pitch_dot, yaw_dot]``.
    leg_pos :
        Per-leg weights for leg lengths.
    leg_vel :
        Per-leg weights for leg velocities.
    leg_ang :
        Per-joint weights for top and bottom joint angles.
        Ordering is ``[top_0..top_5, bot_0..bot_5]``.
    roll :
        Roll weight.
    pitch :
        Pitch weight.
    yaw :
        Yaw weight.
    yaw_ctrl :
        Yaw control weight.
    yaw_dot :
        Yaw velocity weight.
    control :
        Per-axis weights for control effort:
        ``[x_dot2, y_dot2, z_dot2, roll_dot2, pitch_dot2, yaw_dot2]``.
    terminal_exp_scale :
        Exponential scaling factor for terminal-state attenuation.
    terminal_rt_scale :
        Global scale for terminal robot state mismatch term.
    """

    lin_dyn: jax.Array = _init_field(jnp.ones(3))
    omega: jax.Array = _init_field(jnp.ones(3))
    leg_pos: jax.Array = _init_field(jnp.ones(6))
    leg_vel: jax.Array = _init_field(jnp.ones(6))
    leg_ang: jax.Array = _init_field(jnp.ones(12))
    roll: jax.Array = _init_field(jnp.ones(1))
    pitch: jax.Array = _init_field(jnp.ones(1))
    yaw: jax.Array = _init_field(jnp.ones(1))
    yaw_ctrl: jax.Array = _init_field(jnp.ones(1))
    yaw_dot: jax.Array = _init_field(jnp.ones(1))
    control: jax.Array = _init_field(jnp.ones(6))
    terminal_exp_scale: jax.Array = _init_field(jnp.array(10.0))
    terminal_rt_scale: jax.Array = _init_field(jnp.array(0.1))

    def _time_scale(self, n: int, name: str) -> jax.Array:
        """Get time scale weights for flat array.

        See the `ExpWeights` class for a nontrivial implementation
        """
        # identity
        return jnp.ones(n, dtype=float)

    def scale_lin_dyn(self, n: int) -> jax.Array:
        """Get time expanded weights for acceleration cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            2D array of shape ``(n, 3)`` with per-step and per-axis weights.
        """
        time_scale = self._time_scale(n, "acc")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.lin_dyn.size))
        val_scale = jnp.tile(self.lin_dyn, (n, 1))
        return time_scale * val_scale

    def scale_omega(self, n: int) -> jax.Array:
        """Get time expanded weights for angular velocity cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            2D array of shape ``(n, 3)`` with per-step and per-axis weights.
        """
        time_scale = self._time_scale(n, "omega")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.omega.size))
        val_scale = jnp.tile(self.omega, (n, 1))
        return time_scale * val_scale

    def scale_leg_pos(self, n: int) -> jax.Array:
        """Get time expanded weights for leg length cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n * 6,)``.
        """
        time_scale = self._time_scale(n, "leg_pos")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.leg_pos.size))
        val_scale = jnp.tile(self.leg_pos, (n, 1))
        return time_scale * val_scale

    def scale_leg_vel(self, n: int) -> jax.Array:
        """Get time expanded weights for leg velocity cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n * 6,)``.
        """
        time_scale = self._time_scale(n, "leg_vel")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.leg_vel.size))
        val_scale = jnp.tile(self.leg_vel, (n, 1))
        return time_scale * val_scale

    def scale_leg_ang(self, n: int) -> jax.Array:
        """Get time expanded weights for joint angle cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n * 12,)``.
        """
        time_scale = self._time_scale(n, "leg_ang")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.leg_ang.size))
        val_scale = jnp.tile(self.leg_ang, (n, 1))
        return time_scale * val_scale

    def scale_roll(self, n: int) -> jax.Array:
        """Get time expanded weights for roll boundary cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n,)``.
        """
        time_scale = self._time_scale(n, "roll")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.roll.size))
        val_scale = jnp.tile(self.roll, (n, 1))
        return time_scale * val_scale

    def scale_pitch(self, n: int) -> jax.Array:
        """Get time expanded weights for pitch boundary cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n,)``.
        """
        time_scale = self._time_scale(n, "pitch")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.pitch.size))
        val_scale = jnp.tile(self.pitch, (n, 1))
        return time_scale * val_scale

    def scale_yaw(self, n: int) -> jax.Array:
        """Get time expanded weights for yaw boundary cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n,)``.
        """
        time_scale = self._time_scale(n, "yaw")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.yaw.size))
        val_scale = jnp.tile(self.yaw, (n, 1))
        return time_scale * val_scale

    def scale_yaw_ctrl(self, n: int) -> jax.Array:
        """Get time expanded weights for yaw control boundary cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n,)``.
        """
        time_scale = self._time_scale(n, "yaw_ctrl")
        time_scale = jnp.tile(
            time_scale.reshape(-1, 1), (1, self.yaw_ctrl.size)
        )
        val_scale = jnp.tile(self.yaw_ctrl, (n, 1))
        return time_scale * val_scale

    def scale_yaw_dot(self, n: int) -> jax.Array:
        """Get time expanded weights for yaw velocity boundary cost.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n,)``.
        """
        time_scale = self._time_scale(n, "yaw_dot")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.yaw_dot.size))
        val_scale = jnp.tile(self.yaw_dot, (n, 1))
        return time_scale * val_scale

    def scale_control(self, n: int) -> jax.Array:
        """Get time expanded weights for control effort.

        Parameters
        ----------
        n :
            Number of horizon samples.

        Returns
        -------
        scale :
            Flattened weight array of shape ``(n * 7,)``.
        """
        time_scale = self._time_scale(n, "control")
        time_scale = jnp.tile(time_scale.reshape(-1, 1), (1, self.control.size))
        val_scale = jnp.tile(self.control, (n, 1))
        return time_scale * val_scale


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ExpWeights(Weights):
    """Exponential time decaying extension of :class:`Weights`.

    Parameters
    ----------
    alpha_acc :
        Decay rate for accelerations.
    alpha_omega :
        Decay rate for angular velocity.
    alpha_leg_pos :
        Decay rate for leg lengths.
    alpha_leg_vel :
        Decay rate for leg velocity.
    alpha_leg_ang :
        Decay rate for joint angle.
    alpha_roll :
        Decay rate for roll.
    alpha_pitch :
        Decay rate for pitch.
    alpha_yaw :
        Decay rate for yaw.
    alpha_yaw_ctrl :
        Decay rate for yaw control.
    alpha_yaw_dot :
        Decay rate for yaw velocity.
    alpha_control :
        Decay rate for control effort.

    Notes
    -----
    The time profile is ``exp(-k / n * alpha)`` where ``k`` is the
    discrete horizon index and ``n`` is horizon length.
    Namely, ``alpha`` is the maximum exponential decrease factor, or
    alternatively, ``alpha`` is the decay rate when time is normalized to unity.
    """

    alpha_acc: jax.Array = _init_field(jnp.ones(1) * 4.0)
    alpha_omega: jax.Array = _init_field(jnp.ones(1) * 4.0)
    alpha_leg_pos: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_leg_vel: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_leg_ang: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_roll: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_pitch: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_yaw: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_yaw_ctrl: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_yaw_dot: jax.Array = _init_field(jnp.ones(1) * 0.0)
    alpha_control: jax.Array = _init_field(jnp.ones(1) * 0.0)

    def _time_scale(self, n: int, name: str) -> jax.Array:
        """Get time scale weights for flat array."""
        # exponential decrease
        alpha_map = {
            "acc": self.alpha_acc,
            "omega": self.alpha_omega,
            "leg_pos": self.alpha_leg_pos,
            "leg_vel": self.alpha_leg_vel,
            "leg_ang": self.alpha_leg_ang,
            "roll": self.alpha_roll,
            "pitch": self.alpha_pitch,
            "yaw": self.alpha_yaw,
            "yaw_ctrl": self.alpha_yaw,
            "yaw_dot": self.alpha_yaw_dot,
            "control": self.alpha_control,
        }
        return jnp.exp(-jnp.arange(n, dtype=float) / n * alpha_map[name])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CostTerms:
    """Container for the boundary penalties used by the MPC objective.

    Parameters
    ----------
    leg_pos_cost :
        Quartic cost for leg length boundary.
    leg_vel_cost :
        Quartic cost for leg velocity boundary.
    leg_ang_cost :
        Quartic cost for joint angle boundary.
    roll_cost :
        Quartic cost for roll boundary.
    pitch_cost :
        Quartic cost for pitch boundary.
    yaw_cost :
        Quartic cost for yaw boundary.
    yaw_ctrl_cost :
        Quartic cost for yaw controls (non-additive, before filtering).
    yaw_dot_cost :
        Quartic cost for yaw rate boundary.
    """

    leg_pos_cost: quartic_cost.QuarticCost
    leg_vel_cost: quartic_cost.QuarticCost
    leg_ang_cost: quartic_cost.QuarticCost
    roll_cost: quartic_cost.QuarticCost
    pitch_cost: quartic_cost.QuarticCost
    yaw_cost: quartic_cost.QuarticCost
    yaw_ctrl_cost: quartic_cost.QuarticCost
    yaw_dot_cost: quartic_cost.QuarticCost


#######################
# general bookkeeping #
#######################


@dataclasses.dataclass
class MPCLimits:
    """MPC limits (all in SI).

    We list all of the limits on the robot.
    These limits are not directly used in the MPC algorithm.
    Instead, they are used implicitly through constraint functions, e.g.,
    :py:class:`exp_mpc.stewart_min.quartic_cost.QuarticCost`.

    Parameters
    ----------
    leg_min :
        Minimum leg length.
    leg_max :
        Maximum leg length.
    joint_max_angle :
        Maximum joint angle from normal for top and bottom joints.
    max_euler :
        Maximum Euler angle limits for the Stewart platform top.
        Usually used to set the parameters `max_roll`, `max_pitch`, and
        `max_yaw`.
    max_roll :
        Maximum roll angle limit for the Stewart platform top.
    max_pitch :
        Maximum pitch angle limit for the Stewart platform top.
    max_yaw :
        Maximum yaw angle limit for the Stewart platform top.
    max_rotary_yaw :
        Maximum yaw that the rotary top.
    max_leg_vel :
        Maximum allowed leg velocity.
    max_rotary_vel :
        Maximum angular velocity allowed on the rotary top.
    max_cart_table_acc :
        Maximum allowed cartesian acceleration of the table top.
    max_cart_vel :
        Maximum allowed velocity of the human head.
    max_cart_acc :
        Maximum allowed acceleration of the human head.
    max_angle_vel :
        Maximum allowed angular velocity of the human head.
    max_angle_acc :
        Maximum allowed angular acceleration of the human head.
    """

    # robot limits
    leg_min: float = 1160.410000 * 1e-3 + 0.05
    leg_max: float = 1770.010000 * 1e-3 - 0.05
    joint_max_angle: float = float(np.deg2rad(42.0 - 5.0))

    max_euler: float = float(np.deg2rad(35.0))
    max_roll: float = max_euler
    max_pitch: float = max_euler
    max_yaw: float = max_euler
    max_rotary_yaw: float = float(np.deg2rad(90.0))
    max_rotary_yaw_control: float = np.pi

    max_leg_vel: float = 20.0 / 39.37
    max_rotary_vel: float = 0.5
    max_cart_table_acc: float = 8.0

    # human limits
    max_cart_vel: float = 10.0
    max_cart_acc: float = 18.0
    max_angle_vel: float = 4.8
    max_angle_acc: float = 2100.0


@dataclasses.dataclass
class MPCSpec:
    """MPC specification (all in SI).

    MPC parameters that are meant to be statically compiled in an MPC
    simulation.

    Parameters
    ----------
    weights :
        Weight scaling for different cost terms.
    cost_terms :
        Nonlinear cost functions for soft boundary constraints.
    dt :
        Time step for real robot.
    n :
        Horizon length (steps of size `dt`).
    human_displacement :
        Cartesian translation vector from the robot frame to the human head
        frame.
    bots :
        6x3 array of the positions of the bottom joints in the robot frame.
    tops :
        6x3 array of the positions of the top joints in the robot frame.
    bot_normals :
        6x3 array of the normal vectors of the bottom joints.
    top_normals :
        6x3 array of the normal vectors of the top joints.
    cart_home :
        Cartesian translation vector to home.
    tops_home :
        6x3 array of the positions of the top joints in the home configuration.
    lengths_home :
        Scalar of the (average) leg lengths in the home configuration.
    use_rotary :
        True if table has rotary top, and False otherwise.
    vspec_acc :
        Vestibular acceleration model, with integration time step ``dt``.
    vspec_jerk :
        Vestibular jerk model, with integration time step ``dt``.
    vspec_omega :
        Vestibular angular velocity model, with integration time step ``dt``.
    ctrlspec :
        Pre-filtering of all controls, with discretization step ``dt``.
    xyzspec :
        Control filtering for xyz pos, with discretization step ``dt``.
    yspec :
        Control filtering for yaw angle, with discretization step ``dt``.
    tspec :
        Control filtering for tilt quat, with discretization step ``dt``.
    max_iter :
        Maximum L-BFGS iterations.
    max_ls :
        Maximum line search iterations per L-BFGS step.
    unroll :
        Whether to unroll L-BFGS loop (JAX control-flow choice).
    init_norm :
        Initial norm for L-BFGS.
    debug :
        Whether to include debug information in the MPC output.
    use_terminal :
        Whether to include terminal penalties.
    """

    # cost stuff
    weights: tp.Optional[Weights] = None
    cost_terms: tp.Optional[CostTerms] = None

    # control
    dt: float = dt
    n: int = 200

    # robot geometry
    human_displacement: np.ndarray = _default_elem(_human_displacement)
    bots: np.ndarray = _default_elem(_bots)
    tops: np.ndarray = _default_elem(_tops)
    bot_normals: np.ndarray = _default_elem(_bot_normals)
    top_normals: np.ndarray = _default_elem(_top_normals)
    cart_home: np.ndarray = _default_elem(_cart_home)
    tops_home: np.ndarray = _default_elem(_tops_home)
    lengths_home: float = _default_elem(_lengths_home)
    use_rotary: bool = True

    # vestibular specs
    vspec_acc: siso.DiscreteEigSISO = _default_elem(vspec_acc0)
    vspec_jerk: siso.DiscreteEigSISO = _default_elem(vspec_jerk0)
    vspec_omega: siso.DiscreteEigSISO = _default_elem(vspec_omega0)

    # control filtering spec
    ctrlspec: siso.DiscreteSISO = _default_elem(cspec)
    xyzspec: siso.DiscreteSISO = _default_elem(pspec)
    yspec: siso.DiscreteSISO = _default_elem(rspec)
    tspec: siso.DiscreteSISO = _default_elem(pspec)

    # optimization spec
    max_iter: int = 4
    max_ls: int = 2
    unroll: bool = False
    init_norm: float = 1.0
    debug: bool = False
    use_terminal: bool = True

    # WARNING:
    # Our hashing and equality checking are super efficient, but not general.
    # We don't expect the user the make duplicates of MPCSpec.
    # Otherwise, the following code needs to be redefined.

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @classmethod
    def init_weight_margins(
        cls,
        weights: Weights,
        limits: MPCLimits = MPCLimits(),
        margins: list[float] = [0.2, 0.1],
        sizes: list[float] = [1.0, 2**3, 2**8],
        leg_margins: list[float] = [0.3, 0.2, 0.1],
        leg_sizes: list[float] = [1.0, 2**3, 2**5, 2**10],
        euler_margins: list[float] = [0.2 / 3.0, 0.1 / 3.0],
        euler_sizes: list[float] = [1.0, 2**3, 2**8],
        **kwargs,
    ) -> "MPCSpec":
        spec = cls(**kwargs)

        leg_pos_cost = quartic_cost.QuarticCost.from_bounds(
            margins=[0.3, 0.2, 0.1],
            sizes=[1.0, 2**3, 2**5, 2**10],
            low=limits.leg_min,
            high=limits.leg_max,
            center=spec.lengths_home,
        )
        leg_vel_cost = quartic_cost.QuarticCost.from_bounds(
            margins=margins,
            sizes=sizes,
            low=-limits.max_leg_vel,
            high=limits.max_leg_vel,
        )
        leg_ang_cost = quartic_cost.QuarticCost.from_bounds(
            margins=margins,
            sizes=sizes,
            low=-limits.joint_max_angle,
            high=limits.joint_max_angle,
        )
        roll_cost = quartic_cost.QuarticCost.from_bounds(
            margins=euler_margins,
            sizes=euler_sizes,
            low=-limits.max_roll,
            high=limits.max_roll,
        )
        pitch_cost = quartic_cost.QuarticCost.from_bounds(
            margins=euler_margins,
            sizes=euler_sizes,
            low=-limits.max_pitch,
            high=limits.max_pitch,
        )
        yaw_cost = quartic_cost.QuarticCost.from_bounds(
            margins=euler_margins,
            sizes=euler_sizes,
            low=-limits.max_rotary_yaw,
            high=limits.max_rotary_yaw,
        )
        yaw_control_cost = quartic_cost.QuarticCost.from_bounds(
            margins=euler_margins,
            sizes=euler_sizes,
            low=-limits.max_rotary_yaw_control,
            high=limits.max_rotary_yaw_control,
        )
        yaw_dot_cost = quartic_cost.QuarticCost.from_bounds(
            margins=euler_margins,
            sizes=euler_sizes,
            low=-limits.max_rotary_vel,
            high=limits.max_rotary_vel,
        )
        cost_terms = CostTerms(
            leg_pos_cost=leg_pos_cost,
            leg_vel_cost=leg_vel_cost,
            leg_ang_cost=leg_ang_cost,
            roll_cost=roll_cost,
            pitch_cost=pitch_cost,
            yaw_cost=yaw_cost,
            yaw_ctrl_cost=yaw_control_cost,
            yaw_dot_cost=yaw_dot_cost,
        )

        spec.weights = weights
        spec.cost_terms = cost_terms
        return spec
