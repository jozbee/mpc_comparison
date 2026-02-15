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

joint_max_angle = 42.0 * np.pi / 180.0
# top_max_angle = 42.0 * np.pi / 180.0
# bot_max_angle = 42.0 * np.pi / 180.0
max_roll = 35.0 * np.pi / 180.0
max_pitch = 35.0 * np.pi / 180.0
max_yaw = 35.0 * np.pi / 180.0
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
    E0 = sci_lin.expm(ss.A * dt)
    E1 = sci_int.quad_vec(
        f=lambda t: sci_lin.expm(ss.A * (dt - t)) @ ss.B,
        a=0,
        b=dt,
    )[0]
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
    return D, P_inv, EP1, CP


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


# definition
s = ct.tf("s")
acc_transfer = 0.911 * (s + 0.0988)
acc_transfer /= (s + 0.133) * (s + 1.95)
omega_transfer = 10.3 * s * 30 * s
omega_transfer /= (10.2 * s + 1) * (0.1 * s + 1) * (30 * s + 1)

# setup
acc_ss = acc_transfer.to_ss()
omega_ss = omega_transfer.to_ss()
C_acc = acc_ss.C
C_omega = omega_ss.C

# integration matrices
E0_acc, E1_acc = get_E0_E1(acc_ss, dt_sim)
E0_omega, E1_omega = get_E0_E1(omega_ss, dt_sim)

# eigen matrices
D_acc, P_acc_inv, EP1_acc, CP_acc = get_eigen_matrices(E0_acc, E1_acc, C_acc)
D_omega, P_omega_inv, EP1_omega, CP_omega = get_eigen_matrices(
    E0_omega, E1_omega, C_omega
)

# starting conditions
earth_gravity = gravity[2]
moon_gravity = np.array(1.625)  # m / s
fac = acc_transfer(0.0).real
v0_earth = obs_x1(
    E0_acc,
    E1_acc,
    C_acc,
    np.ones(2) * fac * earth_gravity,
    np.ones(1) * earth_gravity,
)
v0_moon = obs_x1(
    E0_acc,
    E1_acc,
    C_acc,
    np.ones(2) * fac * moon_gravity,
    np.ones(1) * moon_gravity,
)
