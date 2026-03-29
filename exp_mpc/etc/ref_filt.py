"""Home for a smoothing spline routine."""

import functools

import scipy.interpolate as sci_interp
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@jax.jit
def line_lstq(g: jax.Array) -> jax.Array:
    m = g.size
    ts = jnp.arange(m, dtype=float)
    tmp = m * (m + 1.0)
    A_inv = jnp.array(
        [
            [12 / ((m - 1) * tmp), -6 / tmp],
            [-6 / tmp, 2 * (2 * m - 1) / tmp],
        ]
    )
    B = jnp.array([jnp.dot(ts, g), jnp.sum(g)])
    a, b = A_inv @ B
    return a * ts + b


@functools.partial(jax.jit, static_argnames=["m"])
def make_L(g: jax.Array, m: int) -> jax.Array:
    n = g.size
    L = jnp.empty(shape=(n - m + 1, m))

    def body(i: jax.Array, L: jax.Array) -> jax.Array:
        gi = jax.lax.dynamic_slice(g, [i], [m])
        L = L.at[i].set(line_lstq(gi))
        return L

    L = jax.lax.fori_loop(0, L.shape[0], body, L)
    return L


@functools.partial(jax.jit, static_argnames=["m"])
def make_Lz(g: jax.Array, m: int) -> jax.Array:
    n = g.size
    L = make_L(g, m)

    def Lz(k: int, ell: int) -> jax.Array:
        idx0 = jnp.clip((k - m + 1) + ell, min=0, max=L.shape[0] - 1)
        idx1 = jnp.clip(m - 1 - ell, min=0, max=L.shape[1] - 1)
        return L[idx0, idx1]

    return jnp.fromfunction(Lz, shape=(n, m), dtype=int)


@functools.partial(jax.jit, static_argnames=["n", "m"])
def make_Hz(n: int, m: int) -> jax.Array:
    def H(k: int, ell: int) -> float:
        zero_cond = (
            (k + ell < m - 1) | (k + ell > n - 1) | (ell < 0) | (ell > m - 1)
        )
        half_cond = (ell == 0) | (ell == m - 1)
        return jax.lax.cond(
            zero_cond,
            lambda: 0.0,
            lambda: jax.lax.cond(
                half_cond,
                lambda: 0.5,
                lambda: 1.0,
            ),
        )

    return jnp.fromfunction(H, shape=(n, m), dtype=int)


@functools.partial(jax.jit, static_argnames=["m"])
def convert_w_y(
    g: jax.Array, dt: jax.Array, m: int
) -> tuple[jax.Array, jax.Array]:
    n = g.size
    Hz = make_Hz(n, m)
    Lz = make_Lz(g, m)
    w = jnp.sum(Hz, axis=1)
    y = jnp.sum(Hz * Lz, axis=1) / w
    return w * dt, y


def causal_smoother(
    data: np.ndarray | jax.Array,
    m: int = 110,
    lam: float = 0.001,
    dt: float = 0.005,
    nu: int | list[int] = 0,
) -> np.ndarray | list[np.ndarray]:
    r"""Filter data using a cuasal least squares method.

    Evaluates a smoothing B-spline :math:`f(t_0),\ldots,f(t_{n - 1})` that
    minimizes a discretized version of a causal least squares functional.
    Here, :math:`n` is the size of the data, :math:`h` is the stepsize `dt`,
    :math:`\Delta t = h \, (m - 1)`, and :math:`T = h \, (n - 1)`.
    The functional to be minimized is given by

    .. math::

        \int_{\Delta t}^T \int_{t - \Delta t}^t
        |\mathcal{L}(g, t - \Delta t, t)(\tau) - f(\tau)|^2 \, \mathrm{d} \tau
        \, \mathrm{d} t + \lambda \, \int_0^T (f''(t))^2 \, \mathrm{d} t

    where :math:`\mathcal{L}: L^2 \times \mathbb{R}^2 \to C^\infty` is the
    least squares operator

    .. math::

        \mathcal{L}(g, s_1, s_2) = \operatorname*{arg\,min}_{L = \mathrm{line}}
        \int_{s_1}^{s_2} |g(t) - L(t)|^2 \, \mathrm{d} t.

    Notice that the causal smoother does not do a good job at inerpolating the initial
    data segment.
    """
    ts = np.arange(data.size) * dt
    w, y = convert_w_y(data, dt, m)
    spline = sci_interp.make_smoothing_spline(ts, y, w=w, lam=lam)
    if isinstance(nu, int):
        return spline(ts, nu=nu)
    elif isinstance(nu, list):
        return [spline(ts, nu=n) for n in nu]
    else:
        raise RuntimeError(f"nu is not int or list, instead {type(nu)}")
