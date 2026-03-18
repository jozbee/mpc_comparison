r"""
We implement a piecewise defined cost function.
We use quartic polynomials in our cost.
Let :math:`[-a, a] \subseteq \mathbb{R}` be a symmetric interval, let
:math:`x_0 = 0 < x_1 < \ldots < x_n = a` denote a partition of :math:`[0, a]`,
and let :math:`y_0 < y_1 < \ldots < y_n` denote corresponding outputs.
Define :math:`p: [-a, a] \to \mathbb{R}` to be a polynomial such that the
following hold.

1. :math:`p(x)` is a quartic polynomial on each interval
2. :math:`p(x) = p(-x)` is symmetric
3. :math:`p(x_k) = y_k` for all :math:`0 \leq k \leq n`
4. :math:`p(x)` is :math:`\mathcal{C}^2([-a, a])`
5. :math:`p'(0) = 0` and :math:`p''(x) \geq 0`
6. the quantity :math:`\int_{-a}^a |p''(x)|^2 \operatorname{d} x` is minimized
   over all polynomials satisfying properties 5--6.

"Of course", such a polynomial should be computed via a minimization routine in
SciPy.
However, the "Of course" was apparently not obvious.
SciPy handled the approximation quite poorly, and it wasn't obvious that
solutions would always exist.
A naive solution is adopted in :class:`QuarticCost`.
This JAX implementation is jit-able and auto-differentiable.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import numpy as np
import jax
import jax.numpy as jnp


########################
# coefficient routines #
########################


def _worst_a1(a4, a3, a2, y, t0, t1):
    """Horizontal line second derivative."""
    return 0.0


def _worst_a0(a4, a3, a2, a1, y, t0, t1):
    """Horizontal line second derivative."""
    return 0.0


def _analytic_a1(a4, a3, a2, y, t0, t1):
    """Enforce second derivative global minimum at knots."""
    return 0.0


def _analytic_a0(a4, a3, a2, a1, y, t0, t1):
    """Enforce endpoint matching."""
    a0 = y - a4 - a3 * (t1 - t0) - 1.0 / 2.0 * a2 * (t1 - t0) ** 2
    # we should include
    #   `a0 -= 1.0 / 6.0 * a1 * (t1 - t0)**3`
    #  but `a1 == 0` by our construction...
    a0 /= (t1 - t0) ** 4 / 12.0
    return a0


def _polyval_special(coeffs: list[float], x: float, t0: float) -> float:
    assert len(coeffs) == 5
    coeffs = [
        1 / 12 * coeffs[0],
        1 / 6 * coeffs[1],
        1 / 2 * coeffs[2],
        coeffs[3],
        coeffs[4],
    ]
    return float(np.polyval(coeffs, x - t0))


def _polyval_specialp(coeffs: list[float], x: float, t0: float) -> float:
    assert len(coeffs) == 5
    coeffs = [
        1 / 3 * coeffs[0],
        1 / 2 * coeffs[1],
        coeffs[2],
        coeffs[3],
    ]
    return float(np.polyval(coeffs, x - t0))


def _polyval_specialpp(coeffs: list[float], x: float, t0: float) -> float:
    assert len(coeffs) == 5
    coeffs = [
        coeffs[0],
        coeffs[1],
        coeffs[2],
    ]
    return float(np.polyval(coeffs, x - t0))


def _quartic_cost(
    knots: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    r"""Quartic cost function.

    Parameters
    ----------
    knots :
        Knot values, starting at zero and strictly increasing.
    y :
        Function values at the knots, starting at zero and strictly increasing.
        These need to increase quickly enough.
        (This can be difficult to check apriori, but it is checked at runtime.)

    Notes
    -----
    We follow som hacky conventions here.
    We define polynomials as a list of coefficients, where
        ``coeffs[k] = [a_0, a_1, a_2, a_3, a_4]``
    which corresponds to the polynomial
        ``p_k(x) = (1/12) a_0 (x - t_k)^4 + (1/6) a_1 (x - t_k)^3 + (1/2) a_2
        (x - t_k)^2 + a_3 (x - t_k) + a_4``
    where ``t_k`` is the knot at index ``k``.
    Namely, we have second derivative
        ``p_k''(x) = a_0 (x - t_k)^2 + a_1 (x - t_k) + a_2``.
    This allows us to associate individual coefficients with continuity
    conditions at the knots.
    The downside is that we have some specialize polyval helper functions
    defined above.

    We also have the following properties:
        1. :math:`C^2`,
        2. :math:`p''(x) >= 0`,
        3. :math:`\int_a^b |p''(x)|^2 dx = is minimizeed.`,
        4. :math:`p(x_k) = y_k` where the :math:`x_k` partition the interval.
    The most interesting condition is :math:`p''(x) >= 0`, which is not
    differentiable when we consider the knot intervals.
    After some thought, this is particularly dangerous when naively coupled with
    curvature minimization requirement.
    We avoid this by requiring :math:`p''''(x) >= 0` everywhere, and we require
    :math:`p'''(x) == 0` at the knots.
    This has a nice visual interpretation.
    It also means that our cost functions have to rapidly increase.
    """
    assert knots.size == y.size
    assert knots.size >= 2
    assert knots[0] == 0.0
    assert y[0] == 0.0
    assert np.all(np.diff(knots) > 0.0)
    assert np.all(np.diff(y) > 0.0)

    # coeffs[k] = [a_0, a_1, a_2, a_3, a_4]
    # p_k(x) = (1/12) a_0 (x - t_k)^4 + (1/6) a_1 (x - t_k)^3 + (1/2) a_2
    #  (x - t_k)^2 + a_3 (x - t_k) + a_4
    a4 = 0.0
    a3 = 0.0
    a2 = 0.0
    a1 = _analytic_a1(a4, a3, a2, y[1], knots[0], knots[1])
    a0 = _analytic_a0(a4, a3, a2, a1, y[1], knots[0], knots[1])
    coeffs_0 = [a0, a1, a2, a3, a4]
    coeffs = [coeffs_0]

    for k in range(1, knots.size - 1):
        a4 = _polyval_special(coeffs[k - 1], knots[k], knots[k - 1])
        a3 = _polyval_specialp(coeffs[k - 1], knots[k], knots[k - 1])
        a2 = _polyval_specialpp(coeffs[k - 1], knots[k], knots[k - 1])

        a1_bad = _worst_a1(a4, a3, a2, y[k + 1], knots[k], knots[k + 1])
        a0_bad = _worst_a0(a4, a3, a2, a1_bad, y[k + 1], knots[k], knots[k + 1])
        coeffs_k_bad = [a0_bad, a1_bad, a2, a3, a4]
        ell_k = _polyval_special(coeffs_k_bad, knots[k + 1], knots[k])
        assert y[k + 1] >= ell_k, f"y[k + 1] = {y[k + 1]}, ell_k = {ell_k}"

        a1 = _analytic_a1(a4, a3, a2, y[k + 1], knots[k], knots[k + 1])
        a0 = _analytic_a0(a4, a3, a2, a1, y[k + 1], knots[k], knots[k + 1])

        coeffs_k = [a0, a1, a2, a3, a4]
        assert a0 >= 0.0, f"a0 = {a0}"
        coeffs.append(coeffs_k)
    coeffs = np.array(coeffs)
    return coeffs


###############
# jax wrapper #
###############


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class QuarticCost:
    """(Jit-able) Quartic cost function.

    Parameters
    ----------
    coeffs :
        Coefficients of the quartic polynomial.
    knots :
        Knot values for the interval :math:`[0, 1]`.
    low :
        Lower bound of the interval.
    high :
        Upper bound of the interval.
    center :
        Center of the cost function.
        Determines a linear scale that may break the symmetry (but not the
        differentiability) of the cost.

    Notes
    -----
    The internal coefficient representation is scaled to the interval
    :math:`[0, 1]`, and we apply a symmetric definition to extend to
    :math:`[-1, 0]`.
    Symmetry is broken by the ``center`` parameter, which is useful if, e.g.,
    the home position of the Stewart platform is not at the center.
    When calling, we linearly the input ``x`` in the interval
    :math:`[\mathrm{low}, \mathrm{high}]` to the interval :math:`[0, 1]`, and
    then we apply the quartic cost function.
    The calling function is jax compatible, i.e., it is differentiable and
    jit-able.

    We recommend using the ``from_bounds`` method for initialization.
    """
    coeffs: jax.Array
    knots: jax.Array
    low: jax.Array
    high: jax.Array
    center: jax.Array

    @classmethod
    def from_bounds(
        cls,
        margins: list[float],
        sizes: list[float],
        low: float,
        high: float,
        center: tp.Optional[float] = None,
    ) -> "QuarticCost":
        """Create a quartic cost function from bounds.

        Parameters
        ----------
        margins :
            Margins for the quartic cost function.
            These are the distances between the knots in the interval
            :math:`[0, 1]`, starting from the right.
            (Idea: ``[0.2, 0.1]`` means we have knots at
            ``[0.0, 0.7, 0.9, 1.0]``.)
        sizes :
            Sizes for the quartic cost function.
            These are the values of the quartic cost function at the nonzero
            knots.
            (Idea: for knots ``[0.0, 0.7, 0.9, 1.0]`` and sizes
            ``[1.0, 2**3, 2**8]``, we match the quartic cost function at the
            knots to the values ``[0.0, 1.0, 2**3, 2**8]``.)
        low :
            Lower bound of the (evaluation) interval.
        high :
            Upper bound of the (evaluation) interval.
        center :
            Center of the cost function.
            Determines a linear scale that may break the symmetry of the cost
            function.
            If ``center`` is given, the function is still twice continuously
            differentiable everywhere.
            If ``None``, the center is set to the midpoint of ``[low, high]``.
        """
        assert len(margins) + 1 == len(sizes)
        assert sorted(sizes) == sizes
        assert sum(margins) <= 1.0
        assert low < high

        if center is not None:
            assert low < center and center < high
        else:
            center = (high + low) / 2.0

        unity_knots = [1.0]
        for m in reversed(margins):
            unity_knots.append(unity_knots[-1] - m)
        unity_knots.append(0.0)
        unity_knots.reverse()
        sizes = [0.0] + sizes
        coeffs = _quartic_cost(
            knots=np.array(unity_knots),
            y=np.array(sizes),
        )
        # an oddity of the return type of quartic cost means that we need to
        #  scale the coefficients (derived from integrating twice)
        scale = np.array([1.0 / 12.0, 1.0 / 6.0, 1.0 / 2.0, 1.0, 1.0])
        coeffs *= np.tile(A=scale, reps=(coeffs.shape[0], 1))
        return cls(
            coeffs=jnp.array(coeffs, dtype=float),
            knots=jnp.array(unity_knots, dtype=float),
            low=jnp.array(low, dtype=float),
            high=jnp.array(high, dtype=float),
            center=jnp.array(center, dtype=float),
        )

    @jax.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the quartic cost function at ``x``.

        To parallelize, use :func:`jax.vmap`.
        """
        assert x.size == 1
        x = jnp.ravel(x).reshape()

        # scale x to [0, 1], where the coefficients are defined

        width = jax.lax.cond(
            x <= self.center,
            lambda: self.center - self.low,
            lambda: self.high - self.center,
        )
        x = jnp.abs(x - self.center) / width

        index = jnp.searchsorted(self.knots, x) - 1
        index = jnp.clip(index, 0, self.knots.size - 2)
        coeffs = self.coeffs.at[index, :].get()
        knot = self.knots.at[index].get()
        return jnp.polyval(coeffs, x - knot)


if __name__ == "__main__":
    # example
    qc = QuarticCost.from_bounds(
        margins=[0.2, 0.1],
        sizes=[1.0, 2**3, 2**8],
        low=-1.0,
        high=3.0,
    )

    import matplotlib.pyplot as plt

    x = jnp.linspace(-1.0, 3.0, 2**18)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, jax.vmap(qc)(x), label="poly")
    # ax.plot(x, jax.vmap(jax.grad((qc)))(x), label="poly")
    # ax.plot(x, jax.vmap(jax.grad(jax.grad((qc))))(x), label="poly")
    plt.show()
