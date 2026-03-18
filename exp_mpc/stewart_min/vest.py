r"""
Our vestibular models are specified componentwise in terms of SISO transfer
functions.
These are used via an integration matrix trick.
First, we specify iteration matrices --- :math:`E_0,E_1` --- for accurate
time-integration, and then we specify a more efficient eigen-value decomposition
method (i.e., specify the diagonal canonical form), which is noticeably more
efficient for gradient back-propogation algorithms.

Fast integration scheme
=======================

We compute the matrix exponential for a small-time step :math:`\Delta t`, and then
iteratively solve the corresponding initial value problem.
We explicitly spell this out.
Given an LTI system

.. math::

  \dot{x} = A x + B u, \quad x(0) = x_0,

the solution is

.. math::

  x(t) = e^{A t} x_0 + \int_0^t e^{A \, (t - \tau)} \, B u(\tau)
  \operatorname{d}\!\tau.

Let :math:`0 = t_0 < t_1 < \ldots < t_N = T` be a uniform partition with each
:math:`\Delta t := t_k - t_{k - 1}` constant.
Suppose that :math:`u(t) \equiv u_k` is a constant on :math:`[t_{k - 1}, t_k]`.
Define :math:`x_k := x(t_k)`, with :math:`x_0` given.
Define the (constant) matrices

.. math::

  E_0 = e^{A \, \Delta t} \quad\text{and}\quad E_1 = \int_0^{\Delta t}
  e^{A \, (\Delta t - \tau)} B \operatorname{d}\!\tau.

(Note that multiplication of :math:`B` in :math:`E_1`.)
Then

.. math::

  x_k = E_0 x_{k - 1} + E_1 u_k, \quad k = 1, \ldots, n.

Faster integration scheme (eigen)
=================================

After implementing vestibular systems in the MPC algorithm, we found a large
increase in computation time for our cost functions.
The back-propogation of gradients was found to be the main culprit.
This was solved by using the diagonal canonical form to improve the efficiency
of the integration scheme.
Consider the recursive problem

.. math::

  x_k = E_0 x_{k - 1} + E_1 u_{k - 1},

with given :math:`u_k` and :math:`x_0`.
We want to update this efficiently, both in the forward pass and in the
backpropagation of gradients.
The following scheme is posited.
Suppose that :math:`E_0` is diagonalizable, say

.. math::

  E_0 = P D P^{-1},

with :math:`D` diagonal and with :math:`P` the corresponding eigenvectors.
If we introduce the change of basis :math:`\tilde{x}_k = P^{-1} x_k` and
:math:`\tilde{u}_k = P^{-1} E_1 u_k`, then we have the update rules

.. math::

  \tilde{x}_k = D \tilde{x}_{k - 1} + \tilde{u}_{k - 1}, \quad
  \tilde{x}_0 = P^{-1} x_0.

again, with :math:`D` diagonal.
So, these update rules can be applied componentwise.
Simply counting floating point operations shows that this is more efficient
(by a constant factor).
More importantly, the back propagation of gradients rule is very simple, because
we are simply acting componentwise in our updates, most of the time.
To get our desired observed variables, we have the conversion

.. math::

  y_k = C P \tilde{x}_k.

Numerical experiments (not committed to git) show that this update
scheme is very stable (for typical horizon lengths).
Numerical experiments also show that this scheme backpropagates gradients about
4 times faster.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import numpy as np
import scipy.linalg as sci_lin
import scipy.integrate as sci_int
import control as ct

import exp_mpc.stewart_min.robo as robo


###########
# helpers #
###########


def get_E0_E1(ss: ct.StateSpace, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Get E0 and E1 integration matrices for VSpec.

    Parameters
    ----------
    ss :
        State space representation.
    dt :
        Integration time step (uniform constant).

    Returns
    -------
    E0 :
        Internal state integration matrix.
    E1 :
        Control integration matrix.
    """
    Z = np.zeros_like(ss.A)
    I = np.eye(*ss.A.shape)  # noqa: E741
    dyn_mat = np.block([[ss.A, Z], [I, Z]])
    y0 = np.block([[I], [Z]])
    E1 = (sci_lin.expm(dyn_mat * dt) @ y0)[ss.A.shape[0] :] @ ss.B

    # check
    E1_int = sci_int.quad_vec(
        f=lambda t: sci_lin.expm(ss.A * (dt - t)) @ ss.B,
        a=0.0,
        b=dt,
    )[0]
    assert np.allclose(E1_int, E1)

    E0 = sci_lin.expm(ss.A * dt)
    return E0, E1


def get_eigen_matrices(
    E0: np.ndarray, E1: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get diagonal canonical form integration matrices.

    Parameters
    ----------
    E0 :
        State integration matrix (non-diagonal).
    E1 :
        Control integration matrix (non-diagonal).
    C :
        `y = C @ x + D @ u`.

    Returns
    -------
    D :
        Eigenvalues of `E0`.
    P :
        Eigenvectors (as columns) of `E0`.
    P_inv :
        Inverse of `P`.
    CP :
        `C @ P`.
    """
    res = sci_lin.eig(E0)
    D, P = res[0], res[1]
    D = D.real
    P_inv = np.linalg.inv(P)
    EP1 = P_inv @ E1
    CP = C @ P
    return D, P, P_inv, EP1, CP


def obs_x1(
    E0: np.ndarray,
    E1: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Returns initial hidden state x1 corresponding to y1.

    Parameters
    ----------
    E0 :
        State integration matrix.
    E1 :
        Control integration matrix.
    C :
        `y = C @ x + D @ u`.
    y :
        Consecutively observed states.
        Assumed to be sampled at constant frequency.
    u :
        Consecutive controls applied.
        We assume that the constant control `u[k]` was applied over the interval
        between `y[k]` and `y[k + 1]`.

    Returns
    -------
    x0 :
        Initial state corresponding to data `y` and `u`.

    Warnings
    --------
    Apparently the current code assumes that `D == 0`, from
    `y = C @ x + D @ u`.
    """
    n = E0.shape[0]
    mpow = np.linalg.matrix_power
    squee = np.squeeze

    y = np.ravel(y)
    u = np.ravel(u)
    assert y.size == n
    assert u.size == n - 1

    # For `n == 3`, we have the following matrices:
    # `
    # O = np.vstack([C, C @ E0, C @ E0 @ E0])
    # U = np.array([
    #     [0., 0.],
    #     [np.squeeze(C @ E1), 0.],
    #     [np.squeeze(C @ E0 @ E1), np.squeeze(C @ E1)],
    # ])
    # `
    # The following code is valid for general `n`.

    O = np.vstack([C @ mpow(E0, i) for i in range(n)])  # noqa: E741
    U_vals = [0.0, 0.0] + [squee(C @ mpow(E0, i) @ E1) for i in range(n - 1)]
    U = np.array(
        [
            [U_vals[j] for j in range(i + 1, i + 1 - (n - 1), -1)]
            for i in range(n)
        ]
    )
    return np.linalg.solve(O, y - U @ u)


def get_V(
    ss: ct.StateSpace,
    state_weight: float,
    control_weight: float,
) -> np.ndarray:
    r"""Returns an LQR terminal cost matrix V.

    Parameters
    ----------
    ss :
        State space.
    state_weight :
        Additional weight to be applied to states.
        See :math:`Q` in the Notes section.
    control_weight :
        Additional weight to be applied to controls.
        See :math:`R` in the Notes section.

    Returns
    -------
    V :
        Quadratic terminal cost matrix.

    Notes
    -----
    Suppose that we have two simultaneous systems:

    .. math::

        \dot{x}(t) &= A \, x(t) + B \, u(t), \quad x(0) = x_0 \\
        y(t) &= C \, x(t) + D \, u(t)

    and
    
    .. math::

        \dot{x_r}(t) &= A \, x_r(t) + B\,  u_r(t), \quad x_r(0) = x_{r, 0} \\
        y_r(t) &= C \, x_r(t) + D \, u_r(t).

    Define the error vectors :math:`x_e(t) := x(t) - x_r(t)` and
    :math:`e(t) := y(t) - y_r(t)`.
    We consider the LQR problem, for
    :math:`Q, R \in \mathbb{R}^{1 \times 1} = \mathbb{R}`:

    .. math::

        J(u) &= \int_0^\infty [e^\top \, Q \, e + u^\top \, R \, u]
        \operatorname*{d}\!t \\
        &= \int_0^\infty [x_e^\top \, C^\top \, Q \, C \, x_e + 2 \, x_e^\top \,
        C^\top \, Q \, D \, u_e + u_e^\top \, D^\top \, Q \, D \, u_e \\
        &\hspace{15em} \mathop{+} u_e^\top \, R \, u_e + 2 \, u_e^\top \, R \,
        u_r + u_r^\top \, R \, u_r] \operatorname*{d}\!t.

    Suppose that :math:`0 < u_r \ll 1` is negligible in the time-to-go value
    function cost (and :math:`u_e` uniformly bounded).
    Then

    .. math::

        J(u) \approx \int_0^\infty [x_e^\top \, Q_e \, x_e + 2 \, x_e^\top \, N_e \,
        u_e + u_e^\top \, R_e \, u_e] \operatorname*{d}\!t

    for

    .. math::

        Q_e = C^\top \, Q \, C, \quad N_e = C^\top \, Q \, D, \quad R_e = D^\top
        \, Q \, D + R.

    Define the value function

    .. math::

        V(x_0) = \min_u J(u).

    Then we define :math:`V` to be the solution to the (generalized) algebraic
    riccati equation to get

    .. math::

        V(x_0) \approx (x_0 - x_{r, 0})^\top \, V (x_0 - x_{r, 0}).
    """
    Q = state_weight * ss.C.T @ ss.C
    Q = (Q + Q.T) / 2.0
    R = state_weight * ss.D.T @ ss.D + control_weight
    N = state_weight * ss.C.T @ ss.D
    _, V, _ = ct.lqr(ss.A, ss.B, Q, R, N)
    return V


###############
# bookkeeping #
###############


@dataclasses.dataclass
class VSpec:
    """Vestibular specification.

    See the module docs :mod:`exp_mpc.stewart_min.vest` for their mathematical
    interpretation.

    Parameters
    ----------
    C :
        ``y = C @ x + D @ u``.
    D :
        ``y = C @ x + D @ u``.
    E0 :
        ``x_k = E0 @ x_{k - 1} + E1 @ u_k``.
    E1 :
        ``x_k = E0 @ x_{k - 1} + E1 @ u_k``.
    eig :
        Eigenvalues of ``E0``.
    P :
        Eigenvectors of ``E0``.
    P_inv :
        Inverse of ``P``.
    EP1 :
        ``P_inv @ E1``.
    CP :
        ``C @ P``.
    V :
        LQR cost matrix (terminal).
    v0_earth :
        Initial hidden state corresponding to steady-state earth gravity.
    v0_moon :
        Initial hidden state corresponding to steady-state moon gravity.
    """

    C: np.ndarray
    D: np.ndarray
    E0: np.ndarray
    E1: np.ndarray
    eig: np.ndarray
    P: np.ndarray
    P_inv: np.ndarray
    EP1: np.ndarray
    CP: np.ndarray
    V: np.ndarray
    v0_earth: tp.Optional[np.ndarray]
    v0_moon: tp.Optional[np.ndarray] = None

    @classmethod
    def transfer2vspec(
        cls,
        transfer: ct.TransferFunction,
        dt: float,
        terminal_weight_state: float = 10.0,
        terminal_weight_control: float = 1.0,
        earth_moon_v0: bool = False,
    ) -> "VSpec":
        """Compute a VSpec from a SISO transfer function.

        Parameters
        ----------
        transfer :
            A SISO transfer function.
        dt :
            Time step for integration matrices.
        terminal_weight_state :
            Tuning for terminal state weighting matrices.
            (Somewhat redundant, cf. :py:func:`exp_mpc.stewart_min.opt.Weights`.
        terminal_weight_control :
            Tuning for terminal control weighting matrices.
            (Somewhat redundant, cf. :py:func:`exp_mpc.stewart_min.opt.Weights`.
        earth_moon_v0 :
            True to compute the internal states at steady-state eart and moon
            gravity.
            Should only be true for SISO transfer functions for specific forces.

        Returns
        -------
        vspec :
            Vestibular specification from a given SISO transfer function.
        """
        ss = transfer.to_ss()
        C = ss.C
        D = ss.D
        E0, E1 = get_E0_E1(ss, dt)
        eig, P, P_inv, EP1, CP = get_eigen_matrices(E0, E1, C)
        V = get_V(ss, terminal_weight_state, terminal_weight_control)

        if earth_moon_v0:
            earth_gravity = robo.gravity[2]
            moon_gravity = robo.moon_gravity[2]
            fac = transfer(0.0).real
            n = E0.shape[0]
            y0 = np.ones(n) * fac
            u0 = np.ones(n - 1)

            v0_earth = obs_x1(E0, E1, C, y0 * earth_gravity, u0 * earth_gravity)
            v0_moon = obs_x1(E0, E1, C, y0 * moon_gravity, u0 * moon_gravity)
        else:
            v0_earth = None
            v0_moon = None

        return cls(C, D, E0, E1, eig, P, P_inv, EP1, CP, V, v0_earth, v0_moon)

    @property
    def n_state(self) -> int:
        """Number of internal states."""
        return self.E0.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


############
# ref defs #
############


s = ct.tf("s")

params = robo.RoboParams()

acc_transfer0 = 0.911 * (s + 0.0988)
acc_transfer0 /= (s + 0.133) * (s + 1.95)
acc_vspec0 = VSpec.transfer2vspec(acc_transfer0, params.dt, earth_moon_v0=True)

omega_transfer0 = 10.3 * s * 30 * s
omega_transfer0 /= (10.2 * s + 1) * (0.1 * s + 1) * (30 * s + 1)
omega_vspec0 = VSpec.transfer2vspec(omega_transfer0, params.dt)

omega_transfer1 = 5.73 * 80 * s**2 * (1 + 0.06 * s)
omega_transfer1 /= (1 + 80 * s) * (1 + 5.73 * s) * (1 + 0.005 * s)
omega_vspec1 = VSpec.transfer2vspec(omega_transfer1, params.dt)

omega_transfer2 = 5.73 * 80 * s**2
omega_transfer2 /= (1 + 80 * s) * (1 + 5.73 * s)
omega_vspec2 = VSpec.transfer2vspec(omega_transfer2, params.dt)

spec_refs: dict[str, tuple[ct.TransferFunction, VSpec]] = {
    "acc0": (acc_transfer0, acc_vspec0),
    "omega0": (omega_transfer0, omega_vspec0),
    "omega1": (omega_transfer1, omega_vspec1),
    "omega2": (omega_transfer2, omega_vspec2),
}
