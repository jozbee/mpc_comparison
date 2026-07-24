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

We compute the matrix exponential for a small-time step :math:`\Delta t`, and
then iteratively solve the corresponding initial value problem.
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

The standard name for this technique is the so called "zero order hold".

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

Note that a diagonal canonical form exists if and only if the poles of the
continuous-time transfer function are unique.
Namely, this does not exist for triple exponential filters of the form
:math:`H(s) = 1 / (1 + s^3)`.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings

import control as ct
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sci_lin

from exp_mpc.stewart_min import comp

##################
# linear algebra #
##################

# the following are routines for precomputation
# namely, they are not written to be jax compatible, but rather more convenient
#  by being more pythonic


def get_eigen_matrices(
    A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get diagonal canonical form integration matrices.

    Parameters
    ----------
    A :
        State integration matrix (non-diagonal).
    B :
        Control integration matrix (non-diagonal).
    C :
        `y = C @ x + D @ u`.

    Returns
    -------
    diag :
        Eigenvalues of `A`.
    P :
        Eigenvectors (as columns) of `A`.
    P_inv :
        Inverse of `P`.
    CP :
        `C @ P`.
    """
    res = sci_lin.eig(A)
    diag, P = res[0], res[1]
    P_vals = sci_lin.svd(P, compute_uv=False)
    if np.any(P_vals < 1e-4):
        warnings.warn(f"P has small sinular values {P_vals}")
    diag = diag.real
    P_inv = np.linalg.inv(P)
    BP = P_inv @ B
    CP = C @ P
    return diag, P, P_inv, BP, CP


def obs_x0(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Returns initial hidden state x0 corresponding to data.

    Parameters
    ----------
    A :
        State integration matrix.
    B :
        Control integration matrix.
    C :
        `y = C @ x + D @ u`.
    D :
        `y = C @ x + D @ u`.
    y :
        Consecutively observed states.
        Assumed to be sampled at constant frequency.
    u :
        Consecutive controls applied.
        We assume that the constant control `u[k + 1]` was applied over the
        interval between `y[k]` and `y[k + 1]`.

    Returns
    -------
    x0 :
        Initial state corresponding to data `y` and `u`.
    """
    n = A.shape[0]
    mpow = np.linalg.matrix_power
    squee = np.squeeze

    y = np.ravel(y)
    u = np.ravel(u)
    assert y.size == n
    assert u.size == n

    if n == 1:
        return np.atleast_1d((y - squee(D) * u) / squee(C))

    # For `n == 3`, we have the following matrices:
    # `
    # O = np.vstack([C, C @ A, C @ A @ A])
    # U = np.array([
    #     [D, 0., 0.],
    #     [0., np.squeeze(C @ B) + D, 0.],
    #     [0., np.squeeze(C @ A @ B), np.squeeze(C @ B) + D],
    # ])
    # `
    # The following code is valid for general `n`.

    O = np.vstack([C @ mpow(A, i) for i in range(n)])

    def U_fun(i, j):
        if i == 0 and j == 0:
            return squee(D)
        elif i == 0 and j != 0:  # noqa: SIM114
            return 0.0
        elif i != 0 and j == 0:  # noqa: SIM114
            return 0.0
        elif j > i:
            return 0.0
        elif i == j:
            return squee(C @ B + D)
        else:
            return squee(C @ mpow(A, i - j) @ B)

    U = np.fromfunction(np.vectorize(U_fun), (n, n), dtype=int)
    return np.linalg.solve(O, y - U @ u)


###############
# integration #
###############


def lti_int(
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: jax.Array,
    x0: jax.Array,
    u: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Fast integration scheme for SISO LTI systems.

    Parameters
    ----------
    A :
        State integration matrix.
    B :
        Control integration matrix.
    C :
        :math:`y = C \, x + D \, u`.
    D :
        :math:`y = C \, x + D \, u`.
    x0 :
        Initial state
    u :
        Control variables.

    Returns
    -------
    x :
        Internal states.
        (Does not contain the initial state.)
    y :
        Observed states.
    """
    x0 = jnp.ravel(x0)
    u = jnp.ravel(u)
    B = jnp.ravel(B)
    C = jnp.ravel(C)
    D = jnp.squeeze(D)

    assert A.shape[0] == A.shape[1]
    assert len(B.shape) == 1
    assert B.size == A.shape[1]
    assert len(C.shape) == 1
    assert len(D.shape) == 0
    assert C.size == A.shape[1]
    assert x0.size == A.shape[1]
    assert u.size > 0

    def scan_body(x0: jax.Array, u: jax.Array) -> jax.Array:
        x1 = A @ x0 + B * u
        return x1, x1

    _, x = jax.lax.scan(scan_body, x0, u)
    y = C @ x.T + D * u
    return x, y


def additive_lti_int(
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: jax.Array,
    x0: jax.Array,
    y0: jax.Array,
    u: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Fast integration scheme for SISO LTI systems with additive controls.

    By additive controls, we mean that the update rule is given by

    .. math::
        x_{k + 1} &= A \, x_k + B \, (y_k + u_k) \\
                  &= (A + B \, C) \, x_k + B \, u_k + B \, D \, u_{k - 1}.
    
    This routine is special because instead of just returning the internal
    states and observed states, we return the actual controls for the system.
    Sometimes, we might want to use the controls :math:`y_k + u_k` as
    constraints elsewhere.


    Parameters
    ----------
    A :
        State integration matrix.
    B :
        Control integration matrix.
    C :
        :math:`y = C \, x + D \, u`.
    D :
        :math:`y = C \, x + D \, u`.
    x0 :
        Initial state.
    y0 :
        Initial (nonadditive) control.
    u :
        Control variables.

    Returns
    -------
    x :
        Internal states.
        (Does not contain the initial state.)
    u0 :
        Non-additive controls that are applied to the filter.
    y :
        Observed states.
    """
    x0 = jnp.ravel(x0)
    u = jnp.ravel(u)
    B = jnp.ravel(B)
    C = jnp.ravel(C)
    D = jnp.squeeze(D)

    assert A.shape[0] == A.shape[1]
    assert len(B.shape) in [0, 1]
    assert B.size == A.shape[1]
    assert len(C.shape) == 1
    assert len(D.shape) == 0
    assert C.size == A.shape[1]
    assert x0.size == A.shape[1], f"{x0.size}, {A.shape[1]}"
    assert u.size > 0

    def scan_body(state, u):
        x0, y0 = state
        u0 = y0 + u
        x1 = A @ x0 + B * u0
        y1 = C @ x1 + D * u0
        return (x1, y1), (x1, u0, y1)

    _, (x, u0, y) = jax.lax.scan(scan_body, (x0, y0), u)
    return x, u0, y


def eigen_lti_int(
    eig: jax.Array,
    BP: jax.Array,
    CP: jax.Array,
    D: jax.Array,
    P_inv: jax.Array,
    P: jax.Array,
    x0: jax.Array,
    u: jax.Array,
    return_eig_states: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """LTI integration, but using eigen-integration matrices.

    Parameters
    ----------
    eig :
        Eigven values for `A`.
    BP :
        `P_inv @ B`.
    CP :
        `C @ P`.
    P_inv :
        Matrix inverse of `P`.
    D :
        `y = C @ P @ x + D * u`.
    x0 :
        Initial (internal) state.
        Needs to be transformed into the
    u :
        Controls.
    return_eig_states :
        If `True`, internal states for the vestibular system are eigen-states.
        If `False`, then internal states are converted back into their original
        basis representation.

    Returns
    -------
    x :
        Internal states (possibly eigen-states).
    y :
        Observed states.

    See Also
    --------
    :mod:`exp_mpc.stewart_min.siso` :
        Specifies the meaning of the integration matrices and the faster eigen
        implementation.
    """
    x0 = P_inv @ x0

    def eig_update(e, b, x0, u):
        x1 = e * x0 + b * u
        return x1, x1

    def eig_scan(e, b, x0):
        eig_update_part = functools.partial(eig_update, e, b)
        _, x = jax.lax.scan(eig_update_part, x0, u)
        return x

    x = jax.vmap(eig_scan)(eig, jnp.ravel(BP), x0).T
    y = CP @ x.T + D * u
    if not return_eig_states:
        x = x @ P.T
    return x, y


def vec_tilt_quat_lti_int(
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: jax.Array,
    x0: jax.Array,
    vec0: jax.Array,
    t0: jax.Array,
    u: jax.Array,
    dt: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Tilt quaternion SISO LTI integration scheme.

    We use the usual componentwise LTI integration scheme on the components of
    the Lie algebra vectors.
    The scheme assumes that we are integrating tilt quaternions, i.e.,
    quaternions without a z-component, i.e.,
    :math:`t = a + b \, i + c \, j + 0 \, k`.
    So, all arithmetic assumes 3-vectors.

    Parameters
    ----------
    A :
        State integration matrix.
    B :
        Control integration matrix.
    C :
        :math:`y = C \, x + D \, u`.
    D :
        :math:`y = C \, x + D \, u`.
    x0 :
        Initial states for all Lie algebra vector components.
    vec0 :
        Initial Lie algebra vector.
    t0 :
        Initial (nonadditive) tilt.
    u :
        Control variables.
    dt :
        Time step for quaternion integration.

    Returns
    -------
    x :
        Internal states.
    q :
        Filtered tilt quaternion states.
    """
    assert len(u.shape) == 2 and u.shape[1] == 2
    assert x0.shape == (2 * A.shape[0],)
    assert vec0.shape == (2,)
    x0 = x0.reshape(-1, A.shape[0])

    tmp_x0, _, tmp_v0 = additive_lti_int(A, B, C, D, x0[0], vec0[0], u[:, 0])
    tmp_x1, _, tmp_v1 = additive_lti_int(A, B, C, D, x0[1], vec0[1], u[:, 1])
    x = jnp.hstack([tmp_x0, tmp_x1])
    v = jnp.transpose(jnp.vstack([tmp_v0, tmp_v1]))
    t = comp.process_tilt(t0, v, dt)
    return x, t


def quat_lti_int(
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: jax.Array,
    x0: jax.Array,
    u: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Unit quaternion SISO LTI integration scheme.

    We use the usual componentwise LTI integration scheme, except at each step,
    the output is normalized to be of unit length.

    Parameters
    ----------
    A :
        State integration matrix.
    B :
        Control integration matrix.
    C :
        :math:`y = C \, x + D \, u`.
    D :
        :math:`y = C \, x + D \, u`.
    x0 :
        Initial states for all quaternion components.
    u :
        Control variables.

    Returns
    -------
    x :
        Internal states.
    q :
        Filtered unit quaternion states.
    """
    B = jnp.ravel(B).reshape(-1, 1)
    C = jnp.ravel(C)
    D = jnp.ravel(D)
    assert x0.shape == (4 * A.shape[0],)
    x0 = x0.reshape(4, -1)

    def quat_update(x0, u):
        u = u.reshape(1, -1)
        x1 = A @ x0 + B @ u
        y1 = jnp.squeeze(C @ x1 + D @ u)
        q1 = y1 / jnp.linalg.norm(y1)
        # noramlizing could, in rare cases, produce nan, but this is rare, so
        #  we ignore
        # note that we return `x1.T` so that the rows represent the quaternion
        #  components
        return x1, (x1.T, q1)

    _, (x, q) = jax.lax.scan(quat_update, x0.T, u)
    return x, q

###############
# bookkeeping #
###############


@dataclasses.dataclass
class DiscreteSISO:
    """Discrete SISO system specification.

    See the module docs :mod:`exp_mpc.stewart_min.siso` for their mathematical
    interpretation.

    Parameters
    ----------
    A :
        `x_k = A @ x_{k - 1} + B @ u_k`.
    B :
        `x_k = A @ x_{k - 1} + B @ u_k`.
    C :
        `y = C @ x + D @ u`.
    D :
        `y = C @ x + D @ u`.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    @classmethod
    def cont2discrete(
        cls,
        transfer: ct.TransferFunction,
        dt: float,
        method: str = "zoh",
    ) -> DiscreteSISO:
        """Compute a discretized SISO statspace system from a transfer function.

        Parameters
        ----------
        transfer :
            A continuous time SISO transfer function.
        dt :
            Time step for integration matrices.
        method :
            Method for discretizing the continuous time SISO system.

        Returns
        -------
        discrete :
            Discretized state space SISO system.
        """
        ss = transfer.to_ss().sample(dt, method=method)
        return cls(ss.A, ss.B, np.squeeze(ss.C), np.squeeze(ss.D))

    @property
    def n_state(self) -> int:
        """Number of internal states."""
        return self.A.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclasses.dataclass
class DiscreteEigSISO(DiscreteSISO):
    """Discrete SISO system specification, with diagonalization.

    See the module docs :mod:`exp_mpc.stewart_min.siso` for their mathematical
    interpretation.

    Parameters
    ----------
    eig :
        Eigenvalues of ``A``.
    P :
        Eigenvectors of ``A``.
    P_inv :
        Inverse of ``P``.
    BP :
        ``P_inv @ B``.
    CP :
        ``C @ P``.
    """

    eig: np.ndarray
    P: np.ndarray
    P_inv: np.ndarray
    BP: np.ndarray
    CP: np.ndarray

    @classmethod
    def cont2discrete(
        cls,
        transfer: ct.TransferFunction,
        dt: float,
    ) -> DiscreteEigSISO:
        """Compute a discretized SISO statspace system from a transfer function.

        Parameters
        ----------
        transfer :
            A continuous time SISO transfer function.
        dt :
            Time step for integration matrices.

        Returns
        -------
        discrete_eig :
            Discretized state space SISO system with eigen decomposition.
        """
        disc = DiscreteSISO.cont2discrete(transfer, dt)
        eig, P, P_inv, BP, CP = get_eigen_matrices(disc.A, disc.B, disc.C)
        return cls(disc.A, disc.B, disc.C, disc.D, eig, P, P_inv, BP, CP)
