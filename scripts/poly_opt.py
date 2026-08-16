import functools
import multiprocessing as mp
import os

import control as ct
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.linalg as sci_lin
import tqdm

from exp_mpc.stewart_min import comp, mpc_spec, opt, siso, utils

jax.config.update("jax_enable_x64", True)
os.environ["PYTHONWARNINGS"] = "ignore"


########
# data #
########


def load_clean_references(file_path: str) -> tuple[jax.Array, jax.Array]:
    data = np.array(pd.read_hdf(file_path))

    # use 100 Hz sampling
    acc_ref = data[::2, 1:4]
    omega_ref = data[::2, 4:]
    assert acc_ref.shape[0] == omega_ref.shape[0]
    assert acc_ref.shape[1] == 3
    assert omega_ref.shape[1] == 3

    # filt (butter)
    s = ct.tf("s") / (2 * np.pi * 5.0)
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

    return acc_ref, omega_ref


def get_data(spec, x_vest, y_vest, acc_ref, omega_ref, idx_vest):
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

    lti_int = functools.partial(siso.lti_int, *ABCD)
    x_data = x_vest[:, x0_slice[0] : x0_slice[1]]
    y_data = y_vest[:, idx_vest]
    ctrl_data = jnp.hstack([acc_ref, omega_ref])[:, idx_vest]
    return lti_int, x_data, y_data, ctrl_data


#######
# opt #
#######


def make_E(spec, alpha):
    alpha = np.eye(3) * -alpha
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    B = np.array([[0], [0], [1]], dtype=float)
    Q = np.diag([1e0, 1e0, 1e0])
    R = np.array([[1e-0]], dtype=float)
    K, _, _ = ct.lqr(A - alpha, B, Q, R)
    E = sci_lin.expm((A - B @ K) * spec.dt)
    return E


def part(arr, size):
    return jnp.fromfunction(
        lambda i, j: arr[i + j], (arr.shape[0] - size, size), dtype=int
    )


def pred_err(spec, pred_fun, data, hist_size):
    t = jnp.arange(1, spec.n + 1, dtype=float) * spec.dt

    def err_body(hist, future, idx):
        pred = pred_fun(hist, t, idx)
        # return jnp.mean(jnp.square((pred - future) * 1e2))
        return jnp.mean(jnp.square((pred - future) * 1e2 * jnp.exp(-0.5 * t)))
        # return jnp.mean(jnp.square((pred - future) * 1e2)[:100])

    hist = part(data, hist_size)
    future = part(data[hist_size:], spec.n)
    hist = hist[: future.shape[0]]
    idx = jnp.arange(hist_size - 1, hist_size - 1 + hist.shape[0])
    err = jax.vmap(err_body)(hist, future, idx)
    return jnp.mean(err)


@functools.partial(jax.jit, static_argnames=["spec"])
def cost_fun(spec, y_data, E, n_taylor):
    mixed_pred = functools.partial(comp.pred_hist, n_taylor, spec.n, spec.dt, E)
    err = pred_err(spec, lambda hist, _, __: mixed_pred(hist), y_data, 4)
    return err


@functools.partial(jax.jit, static_argnames=["spec", "lti_int"])
def const_cost_fun(spec, lti_int, x_data, y_data, ctrl_data):
    def pred_fun_check(hist, t, idx):
        x0 = x_data[idx]
        _, y = lti_int(x0=x0, u=jnp.ones_like(t) * ctrl_data[idx])
        return y

    return pred_err(spec, pred_fun_check, y_data, 2)


def poly_opt(spec, lti_int, x_data, y_data, ctrl_data):
    res = {}
    alphas = np.arange(0.2, 22.0, step=0.05)
    n_taylors = range(0, 10, 2)
    lti_int = jax.jit(lti_int)

    for j in tqdm.tqdm(range(len(n_taylors))):
        for i in range(len(alphas)):
            E_i = make_E(spec, alphas[i])
            err = cost_fun(spec, y_data, E_i, n_taylors[j])
            res[(i, j)] = err

    s_res = sorted([(elem, key) for key, elem in list(res.items())])
    return {
        "alpha": float(np.round(alphas[s_res[0][1][0]], decimals=2)),
        "n_taylor": n_taylors[s_res[0][1][1]],
        "err": float(s_res[0][0]),
        "const_err": float(
            const_cost_fun(spec, lti_int, x_data, y_data, ctrl_data)
        ),
    }


########
# main #
########


if __name__ == "__main__":
    # data
    file_paths = [
        "../data/clean_00_sms_drive.hdf",
        "../data/clean_specific-forces-standard-road-v2.hdf",
        "../data/clean_specific-forces-lander_motion_redes_manual.hdf",
        "../data/clean_specific-forces-lander_motion_redes_auto.hdf",
    ]
    file_path = file_paths[2]
    acc_ref, omega_ref = load_clean_references(file_path)

    # setup
    spec = mpc_spec.MPCSpec()
    ts = opt.TrainState.zero_init(spec, sim_acc_z=acc_ref[0, 2])
    eigen_vstates = jax.jit(
        utils.eigen_vstates, static_argnames=["spec", "return_eig_states"]
    )
    x_vest, y_vest = eigen_vstates(
        spec=spec,
        acc=acc_ref,
        omega=omega_ref,
        vstate0=ts.vstate0_sim,
        return_eig_states=False,
    )

    # run
    poly_opt_args = []
    for data_idx in range(6):
        data = get_data(spec, x_vest, y_vest, acc_ref, omega_ref, data_idx)
        poly_opt_args.append([spec, *data])

    with mp.Pool(processes=6) as pool:
        res = pool.starmap(poly_opt, poly_opt_args)

    # print
    print(f"file_path = {file_path}")
    for i in range(len(res)):
        print(f"{i}: {res[i]}")
