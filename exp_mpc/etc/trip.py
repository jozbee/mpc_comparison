"""Routines for triple exponential filter."""

import functools
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def fast_trip_E0(f: jax.Array) -> jax.Array:
    x0 = f**2
    x1 = x0 + 80000
    x2 = jnp.exp((1 / 200) * f)
    x3 = (1 / 80000) * x2
    x4 = (1 / 40000) * x2
    x5 = f**3
    x6 = x3 * (f + 400)
    return jnp.array(
        [
            [x3 * (800 * f + x1), x0 * x4 * (-f - 600), x5 * x6],
            [x6, x4 * (-200 * f - x0 + 40000), x3 * x5],
            [x3, x4 * (200 - f), x3 * (-400 * f + x1)],
        ]
    )


def fast_trip_E1(f: jax.Array) -> jax.Array:
    x0 = (1 / 200) * f
    x1 = jnp.exp(x0)
    x2 = (1 / 80000) * x1
    return jnp.array(
        [[x2 * (f + 400)], [x2], [(f**2 * x2 - x0 * x1 + x1 - 1) / f**3]]
    )


def fast_trip_C(f: jax.Array, nu: int) -> jax.Array:
    assert 0 <= nu and nu <= 2
    if nu == 0:
        return jnp.array([0, 0, (-f) ** 3])
    elif nu == 1:
        return jnp.array([0, (-f) ** 3, 0])
    else:  # nu == 2
        return jnp.array([(-f) ** 3, 0, 0])


@functools.partial(jax.jit, static_argnames=["nu"])
def fast_trip_E0_E1_C(
    f: jax.Array, nu: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return fast_trip_E0(f), jnp.ravel(fast_trip_E1(f)), fast_trip_C(f, nu)

@jax.jit
def fast_trip_E0_E1_C_full(
    f: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    E0 = fast_trip_E0(f)
    E1 = jnp.ravel(fast_trip_E1(f))
    C0 = fast_trip_C(f, 0)
    C1 = fast_trip_C(f, 1)
    C2 = fast_trip_C(f, 2)
    return E0, E1, C0, C1, C2


@jax.jit
def fast_obs_x0(f, y_0, y_1, y_2, u_0, u_1):
    x0 = f**3
    x1 = x0 ** (-1.0)
    x2 = f * y_2
    x3 = f**2
    x4 = jnp.exp((1 / 200) * f)
    x5 = jnp.exp((1 / 100) * f)
    x6 = x5 * y_0
    x7 = 160000 * x4
    x8 = f * u_0
    x9 = x3 * x4
    return jnp.ravel(
        jnp.array(
            [
                [
                    (1 / 400)
                    * x1
                    * (
                        80000 * f * u_0 * x5
                        - f * u_1 * x7
                        + 240000 * f * u_1
                        + 320000 * f * x4 * y_1
                        - 80000 * f * x6
                        - u_0 * x0 * x4
                        - 16000000 * u_0 * x4
                        + 16000000 * u_0 * x5
                        - 600 * u_0 * x9
                        + u_1 * x0 * x4
                        + 600 * u_1 * x3 * x4
                        + 400 * u_1 * x3
                        - 16000000 * u_1 * x4
                        + 16000000 * u_1
                        - 240000 * x2
                        - 400 * x3 * y_2
                        + 32000000 * x4 * y_1
                        - 16000000 * x6
                        - x7 * x8
                        - 16000000 * y_2
                    )
                ],
                [
                    x1
                    * (
                        (1 / 2) * f * u_1 * x4
                        + f * u_1
                        - 100 * u_0 * x4
                        + 100 * u_0 * x5
                        - 1 / 800 * u_0 * x9
                        + (1 / 800) * u_1 * x3 * x4
                        - 300 * u_1 * x4
                        + 300 * u_1
                        - x2
                        - 1 / 2 * x4 * x8
                        + 400 * x4 * y_1
                        - 100 * x6
                        - 300 * y_2
                    )
                ],
                [-x1 * y_2],
            ]
        )
    )
