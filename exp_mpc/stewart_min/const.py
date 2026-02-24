"""Constants used in Stewart platform MPC specification."""

from __future__ import annotations

import dataclasses
import numpy as np
import scipy.linalg as sci_lin
import scipy.integrate as sci_int
import control as ct


###############
# constraints #
###############


leg_min = 1160.410000 * 1e-3
leg_max = 1770.010000 * 1e-3
leg_mid = (leg_min + leg_max) / 2.0  # 1.46521
dt = 0.005
# dt_sim = dt
dt_sim = dt * 2.0  # WARNING: only multiply by a number >= 1

joint_max_angle = 42.0 * np.pi / 180.0
# top_max_angle = 42.0 * np.pi / 180.0
# bot_max_angle = 42.0 * np.pi / 180.0
max_euler = 35.0 * np.pi / 180.0
max_roll = max_euler
max_pitch = max_euler
max_yaw = max_euler
max_rotary_yaw = 90.0 * np.pi / 180.0
max_leg_vel = 20.0 / 39.37
max_rotary_vel = 0.5  # rad / s
max_cart_table_acc = 8.0
max_cart_vel = 10.0  # human
max_cart_acc = 18.0  # human
max_angle_vel = 4.8  # human
max_angle_acc = 2100.0  # human


########
# misc #
########


# warning: positive
# (think: to stay in place on earth, we are accelerating up wards to counteract
#   gravity)
gravity = np.array([0.0, 0.0, 9.81])
human_displacement = np.array([0.0, 0.0, 0.588])


############
# geometry #
############


bots = np.array(
    [
        [952.5055, 91.0723, -1410.0000],
        [-398.5396, 869.5826, -1409.8621],
        [-555.4801, 779.1038, -1410.0000],
        [-555.0219, -779.3507, -1409.6010],
        [-398.5396, -869.9006, -1410.0000],
        [952.7381, -89.7865, -1409.8718],
    ]
)
bots *= 1e-3
tops = np.array(
    [
        [314.4868, 327.8608, -111.0000],
        [126.7447, 436.2739, -111.1102],
        [-441.2953, 107.9497, -111.0000],
        [-441.2826, -108.6562, -111.3975],
        [126.7447, -436.2688, -111.0000],
        [314.5916, -328.3827, -110.9675],
    ]
)
tops *= 1e-3

top_normals = np.array(
    [
        [0.435014, -0.162005, -0.885729],
        [-0.357803, 0.295803, -0.885708],
        [-0.077200, 0.457799, -0.885698],
        [-0.077200, -0.457799, -0.885698],
        [-0.357803, -0.295803, -0.885708],
        [0.435014, 0.162005, -0.885729],
    ]
)
top_normals /= np.linalg.norm(top_normals, axis=1)[:, np.newaxis]
bot_normals = np.array(
    [
        [-0.435014, 0.162005, 0.885729],
        [0.357803, -0.295803, 0.885708],
        [0.077200, -0.457799, 0.885698],
        [0.077200, 0.457799, 0.885698],
        [0.357803, 0.295803, 0.885708],
        [-0.435014, -0.162005, 0.885729],
    ]
)
bot_normals /= np.linalg.norm(bot_normals, axis=1)[:, np.newaxis]


####################
# vestibular model #
####################

# see `vestibular.ipynb` for the interpretation of the following quantities


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


@dataclasses.dataclass
class VSpec:
    """Vestibular specification."""

    C: np.ndarray
    D: np.ndarray
    E0: np.ndarray
    E1: np.ndarray
    eig: np.ndarray
    P: np.ndarray
    P_inv: np.ndarray
    EP1: np.ndarray
    CP: np.ndarray
    v0_earth: np.ndarray
    v0_moon: np.ndarray

    @classmethod
    def transfer2vspec(
        cls, transfer: ct.TransferFunction, dt: float
    ) -> "VSpec":
        ss = transfer.to_ss()
        C = ss.C
        D = ss.D
        E0, E1 = get_E0_E1(ss, dt)
        eig, P, P_inv, EP1, CP = get_eigen_matrices(E0, E1, C)

        earth_gravity = gravity[2]
        moon_gravity = np.array(1.625)  # m / s
        fac = acc_transfer(0.0).real
        n = E0.shape[0]
        y0 = np.ones(n) * fac
        u0 = np.ones(n - 1)

        # (admittedly, these the v0_earth and v0_moon are not really meaningful
        #  in the omega case...)
        v0_earth = obs_x1(E0, E1, C, y0 * earth_gravity, u0 * earth_gravity)
        v0_moon = obs_x1(E0, E1, C, y0 * moon_gravity, u0 * moon_gravity)

        return cls(C, D, E0, E1, eig, P, P_inv, EP1, CP, v0_earth, v0_moon)


# definition
s = ct.tf("s")
acc_transfer = 0.911 * (s + 0.0988)
acc_transfer /= (s + 0.133) * (s + 1.95)
omega_transfer = 10.3 * s * 30 * s
omega_transfer /= (10.2 * s + 1) * (0.1 * s + 1) * (30 * s + 1)
# omega_transfer = 5.73 * 80 * s**2 * (1 + 0.06 * s)
# omega_transfer /= (1 + 80 * s) * (1 + 5.73 * s) * (1 + 0.005 * s)
# omega_transfer = 5.73 * 80 * s**2
# omega_transfer /= (1 + 80 * s) * (1 + 5.73 * s)

# vspec
vspec_acc = VSpec.transfer2vspec(acc_transfer, dt)
vspec_omega = VSpec.transfer2vspec(omega_transfer, dt)
vspec_acc_sim = VSpec.transfer2vspec(acc_transfer, dt_sim)
vspec_omega_sim = VSpec.transfer2vspec(omega_transfer, dt_sim)

##########################
# terminal cost modeling #
##########################

ome_ss = omega_transfer.to_ss()
acc_ss = acc_transfer.to_ss()

ome_Q = 10.0 * ome_ss.C.T @ ome_ss.C
ome_R = 10.0 * ome_ss.D.T @ ome_ss.D + 1.0
ome_N = 10.0 * ome_ss.C.T @ ome_ss.D

acc_Q = 10.0 * acc_ss.C.T @ acc_ss.C
acc_R = 10.0 * acc_ss.D.T @ acc_ss.D + 1.0
acc_N = 10.0 * acc_ss.C.T @ acc_ss.D

z_acc_Q = 1.0 * acc_ss.C.T @ acc_ss.C
z_acc_R = 1.0 * acc_ss.D.T @ acc_ss.D + 1.0
z_acc_N = 1.0 * acc_ss.C.T @ acc_ss.D

_, ome_V, _ = ct.lqr(ome_ss.A, ome_ss.B, ome_Q, ome_R, ome_N)
_, acc_V, _ = ct.lqr(acc_ss.A, acc_ss.B, acc_Q, acc_R, acc_N)
_, z_acc_V, _ = ct.lqr(acc_ss.A, acc_ss.B, z_acc_Q, z_acc_R, z_acc_N)
