import functools

import control as ct
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import tqdm

from exp_mpc.stewart_min import mpc_spec, opt, siso, utils

jax.config.update("jax_enable_x64", True)

########
# data #
########


# load data
def load_clean_references(file_path: str) -> tuple[jax.Array, jax.Array]:
    data = np.array(pd.read_hdf(file_path))
    return data[:, 1:4], data[:, 4:]


# file_path = "../../data/clean_00_sms_drive.hdf"
# file_path = "../../data/clean_specific-forces-standard-road-v2.hdf"
file_path = "../../data/clean_specific-forces-lander_motion_redes_manual.hdf"
# file_path = "../../data/clean_specific-forces-lander_motion_redes_auto.hdf"
acc_ref, omega_ref = load_clean_references(file_path)

# use 100 Hz sampling
acc_ref = acc_ref[::2]
omega_ref = omega_ref[::2]

assert acc_ref.shape[0] == omega_ref.shape[0]
assert acc_ref.shape[1] == 3
assert omega_ref.shape[1] == 3

# filt (butter)
s = ct.tf("s") / (2 * np.pi * 0.5)
butter = siso.DiscreteSISO.cont2discrete(
    1 / (1 + 2 * s + 2 * s**2 + s**3), dt=mpc_spec.dt, method="bilinear"
)
ABCD = [butter.A, butter.B, butter.C, butter.D]


def butter_int(u):
    assert len(u.shape) == 1
    one = np.ones(ABCD[0].shape[0])
    x0 = siso.obs_x0(*ABCD, y=one * u[0], u=one * u[0])
    return siso.lti_int(*ABCD, x0, u)[1]


acc_ref = jnp.array([butter_int(u) for u in acc_ref.T]).T
omega_ref = jnp.array([butter_int(u) for u in omega_ref.T]).T

spec = mpc_spec.MPCSpec()
ts = opt.TrainState.zero_init(spec, sim_acc_z=acc_ref[0, 2])
x_vest, y_vest = jax.jit(
    utils.eigen_vstates, static_argnames=["spec", "return_eig_states"]
)(
    spec=spec,
    acc=acc_ref,
    omega=omega_ref,
    vstate0=ts.vstate0_sim,
    return_eig_states=False,
)

####################
# helper functions #
####################


def part(arr, size):
    return jnp.fromfunction(
        lambda i, j: arr[i + j], (arr.shape[0] - size, size), dtype=int
    )


def pred_err(pred_fun, data, hist_size):
    t = jnp.arange(1, spec.n + 1, dtype=float) * spec.dt

    def err_body(hist, future, idx):
        pred = pred_fun(hist, t, idx)
        return jnp.mean(jnp.square((pred - future) * 1e2))
        # return jnp.mean(jnp.square((pred - future) * 1e2 * jnp.exp(-1.0 * t)))
        # return jnp.mean(jnp.square((pred - future) * 1e2)[:100])

    hist = part(data, hist_size)
    future = part(data[hist_size:], spec.n)
    hist = hist[: future.shape[0]]
    idx = jnp.arange(hist_size - 1, hist_size - 1 + hist.shape[0])
    err = jax.vmap(err_body)(hist, future, idx)
    return jnp.mean(err)

def get_data(idx_vest):
    assert 0 <= idx_vest < 6
    if idx_vest < 2:
        vspec = spec.vspec_acc
        x0_slice = [idx_vest * vspec.n_state, (idx_vest + 1) * vspec.n_state]
    elif idx_vest == 2:
        vspec = spec.vspec_jerk
        start_idx = spec.vspec_acc.n_state * 2
        x0_slice = [start_idx, start_idx + vspec.n_state]
    else:  # idx_vest < 6:
        vspec = spec.vspec_omega
        start_idx = spec.vspec_acc.n_state * 2 + spec.vspec_jerk.n_state
        x0_slice = [
            start_idx + vspec.n_state * (idx_vest - 3),
            start_idx + vspec.n_state * (idx_vest + 1 - 3),
        ]

    ABCD = [vspec.A, vspec.B, vspec.C, vspec.D]

    lti_int = jax.jit(functools.partial(siso.lti_int, *ABCD))
    x_data = x_vest[:, x0_slice[0]: x0_slice[1]]
    y_data = y_vest[:, idx_vest]
    ctrl_data = jnp.hstack([acc_ref, omega_ref])[:, idx_vest]
    return lti_int, x_data, y_data, ctrl_data
