"""We implement a custom quartic cost function.

Given an interval [a, b], we define a symmetric piecewise quartic polynomial
that has the properties:

1. C^2,
2. p''(x) >= 0,
3. \\int_a^b |p(x)|^2 dx = is minimizeed.
4. p(x_k) = y_k where the x_k partition the interval.

Technically, we restrict our attention to $y_k$ that are particularly nice,
e.g., y_{k + 1} >> y_k.
This is a useful cost function that essentially "blows up", while still allowing
optimization routines to work.
"""

from __future__ import annotations

import dataclasses

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
    a0 = y - a4 - a3 * (t1 - t0) - 1.0 / 2.0 * a2 * (t1 - t0)**2
    # we should include
    #   `a0 -= 1.0 / 6.0 * a1 * (t1 - t0)**3`
    #  but `a1 == 0` by our construction...
    a0 /= (t1 - t0)**4 / 12.0
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
    """Quartic cost function.

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
        `coeffs[k] = [a_0, a_1, a_2, a_3, a_4]`
    which corresponds to the polynomial
        `p_k(x) = (1/12) a_0 (x - t_k)^4 + (1/6) a_1 (x - t_k)^3 + (1/2) a_2 (x - t_k)^2 + a_3 (x - t_k) + a_4`
    where `t_k` is the knot at index `k`.
    Namely, we have second derivative
        `p_k''(x) = a_0 (x - t_k)^2 + a_1 (x - t_k) + a_2`.
    This allows us to associate individual coefficients with continuity
    conditions at the knots.
    The downside is that we have some specialize polyval helper functions
    defined above.

    We also have the following properties:
        1. C^2,
        2. p''(x) >= 0,
        3. \\int_a^b |p''(x)|^2 dx = is minimizeed.
        4. p(x_k) = y_k where the x_k partition the interval.
    The most interesting condition is p''(x) >= 0, which is not differentiable
    when we consider the knot intervals.
    After some thought, this is particularly dangerous when naively coupled with
    curvature minimization requirement.
    We avoid this by requiring p''''(x) >= 0 everywhere, and we require
    p'''(x) == 0 at the knots.
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
    # p_k(x) = (1/12) a_0 (x - t_k)^4 + (1/6) a_1 (x - t_k)^3 + (1/2) a_2 (x - t_k)^2 + a_3 (x - t_k) + a_4
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

    The internal coefficient representation is scaled to the interval [0, 1],
    and we apply a symmetric definition to extend to [-1, 0].
    When calling, we linearly the input x in the interval [low, high] to the
    interval [0, 1], and we apply the quartic cost function.
    The calling function is differentiable and jit-able from jax.
    Use the `from_bounds` method for initialization.

    Attributes
    ----------
    coeffs :
        Coefficients of the quartic polynomial.
    knots :
        Knot values for the interval [0, 1].
    low :
        Lower bound of the interval.
    high :
        Upper bound of the interval.
    """
    coeffs: jax.Array
    knots: jax.Array
    low: jax.Array
    high: jax.Array

    @classmethod
    def from_bounds(
        cls,
        margins: list[float],
        sizes: list[float],
        low: float,
        high: float,
    ) -> "QuarticCost":
        """Create a quartic cost function from bounds.
        
        Parameters
        ----------
        margins :
            Margins for the quartic cost function.
            These are the distances between the knots in the interval [0, 1],
            starting from the right.
            (Idea: [0.2, 0.1] means we have knots at [0.0, 0.7, 0.9, 1.0].)
        sizes :
            Sizes for the quartic cost function.
            These are the values of the quartic cost function at the nonzero
            knots.
            (Idea: for knots [0.0, 0.7, 0.9, 1.0] and sizes [1.0, 2**3, 2**8],
            we match the quartic cost function at the knots to the values
            [0.0, 1.0, 2**3, 2**8].)
        low :
            Lower bound of the (evaluation) interval.
        high :
            Upper bound of the (evaluation) interval.
        """
        assert len(margins) + 1 == len(sizes)
        assert sorted(sizes) == sizes
        assert sum(margins) <= 1.0
        assert low < high

        unity_knots = [1.0]
        for m in margins:
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
        )

    @jax.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the quartic cost function at x.
        
        To parallelize, use `jax.vmap`.
        """
        assert x.size == 1
        x = jnp.ravel(x).reshape()

        # scale x to [0, 1], where the coefficients are defined
        mid = (self.high + self.low) / 2.0
        width = (self.high - self.low) / 2.0
        x = jnp.abs(x - mid) / width

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
