r"""Vestibular model implementation.

Our vestibular models are specified componentwise in terms of SISO transfer
functions.
These are used via an integration matrix trick.
First, we specify interation matrices --- :math:`E_0,E_1` --- for accurate
time-integration, and then we specify a more efficient eigen-value decomposition
method, which is noticeably more efficient for back-propogation algorithms.

## Fast integration scheme

Compute the matrix exponential for a small-time step :math:`\Delta t`, and then
iteratively solve the corresponding initial value problem.
We explicitly spell this out.
Given an LTI system

.. math::

  \dot{x} = A x + B u, \quad x(0) = x_0,

the solution is

.. math::

  x(t) = e^{A t} x_0 + \int_0^t e^{A \, (t - \tau)} \, B u(\tau)
  \operatorname{d}\!\tau.

Let :math:`0 = t_0 < t_1 < \ldots < t_N = T` be a partition with each difference
:math:`t_k - t_{k - 1} = \Delta t` constant.
Suppose that :math:`u(t) \equiv u_k` is a constant on :math:`[t_{k - 1}, t_k]`.
Define :math:`x_k := x(t_k)`, where :math:`x_0` is given.
Define the (constant) matrices

.. math::

  E_0 = e^{A \, \Delta t} \quad\text{and}\quad E_1 = \int_0^{\Delta t}
  e^{A \, (\Delta t - \tau)} B \operatorname{d}\!\tau.

(Note that multiplication of :math:`B` in :math:`E_1`.)
Then

.. math::

  x_k = E_0 x_{k - 1} + E_1 u_k, \quad k = 1, \ldots, n.

## Faster integration scheme (eigen)

We introduce fancy linear-algebraic update scheme.
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
If we introduce the notation :math:`\tilde{x}_k = P^{-1} x_k` and
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

Numerical experiments (done in a temporary notebook) show that this update
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
import jax
import control as ct

import exp_mpc.stewart_min.const as const


###########
# helpers #
###########


def _static_field() -> tp.Any:
    return dataclasses.field(metadata=dict(static=True))


# def _dyn_field() -> tp.Any:
#     return dataclasses.field()


def get_E0_E1(ss: ct.StateSpace, dt: float) -> tuple[np.ndarray, np.ndarray]:
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
    """Returns initial hidden state x1 corresponding to y1."""
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
    """Returns an LQR cost matrix V."""
    Q = state_weight * ss.C.T @ ss.C
    Q = (Q + Q.T) / 2.0
    R = state_weight * ss.D.T @ ss.D + control_weight
    N = state_weight * ss.C.T @ ss.D
    _, V, _ = ct.lqr(ss.A, ss.B, Q, R, N)
    return V


###############
# bookkeeping #
###############


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class VSpec:
    """Vestibular specification.
    
    C :
        `y = C @ x + D @ u`
    D :
        `y = C @ x + D @ u`
    E0 :
        `x_k = E0 @ x_{k - 1} + E1 @ u_k`
    E1 :
        `x_k = E0 @ x_{k - 1} + E1 @ u_k`
    eig :
        Eigenvalues of E0.
    P :
        Eigenvectors of E0.
    P_inv :
        Inverse of P.
    EP1 :
        P_inv @ E1.
    CP :
        C @ P.
    V :
        LQR cost matrix (terminal).
    v0_earth :
        Initial hidden state corresponding to steady-state earth gravity.
    v0_moon :
        Initial hidden state corresponding to steady-state moon gravity.
    """

    C: np.ndarray = _static_field()
    D: np.ndarray = _static_field()
    E0: np.ndarray = _static_field()
    E1: np.ndarray = _static_field()
    eig: np.ndarray = _static_field()
    P: np.ndarray = _static_field()
    P_inv: np.ndarray = _static_field()
    EP1: np.ndarray = _static_field()
    CP: np.ndarray = _static_field()
    V: np.ndarray = _static_field()
    v0_earth: tp.Optional[np.ndarray] = _static_field()
    v0_moon: tp.Optional[np.ndarray] = _static_field()

    @classmethod
    def transfer2vspec(
        cls,
        transfer: ct.TransferFunction,
        dt: float,
        terminal_weight_state: float = 10.0,
        terminal_weight_control: float = 1.0,
        earth_moon_v0: bool = False,
    ) -> "VSpec":
        ss = transfer.to_ss()
        C = ss.C
        D = ss.D
        E0, E1 = get_E0_E1(ss, dt)
        eig, P, P_inv, EP1, CP = get_eigen_matrices(E0, E1, C)
        V = get_V(ss, terminal_weight_state, terminal_weight_control)

        if earth_moon_v0:
            earth_gravity = const.gravity[2]
            moon_gravity = const.moon_gravity[2]
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
        return self.E0.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


############
# ref defs #
############


s = ct.tf("s")

acc_transfer0 = 0.911 * (s + 0.0988)
acc_transfer0 /= (s + 0.133) * (s + 1.95)
acc_vspec0 = VSpec.transfer2vspec(acc_transfer0, const.dt, earth_moon_v0=True)

omega_transfer0 = 10.3 * s * 30 * s
omega_transfer0 /= (10.2 * s + 1) * (0.1 * s + 1) * (30 * s + 1)
omega_vspec0 = VSpec.transfer2vspec(omega_transfer0, const.dt)

omega_transfer1 = 5.73 * 80 * s**2 * (1 + 0.06 * s)
omega_transfer1 /= (1 + 80 * s) * (1 + 5.73 * s) * (1 + 0.005 * s)
omega_vspec1 = VSpec.transfer2vspec(omega_transfer1, const.dt)

omega_transfer2 = 5.73 * 80 * s**2
omega_transfer2 /= (1 + 80 * s) * (1 + 5.73 * s)
omega_vspec2 = VSpec.transfer2vspec(omega_transfer2, const.dt)

spec_refs: dict[str, tuple[ct.TransferFunction, VSpec]] = {
    "acc0": (acc_transfer0, acc_vspec0),
    "omega0": (omega_transfer0, omega_vspec0),
    "omega1": (omega_transfer1, omega_vspec1),
    "omega2": (omega_transfer2, omega_vspec2),
}
