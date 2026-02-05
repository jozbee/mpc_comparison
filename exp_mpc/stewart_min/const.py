import numpy as np
import scipy.linalg as sci_lin


leg_min = 1160.410000 * 1e-3
leg_max = 1770.010000 * 1e-3
leg_mid = (leg_min + leg_max) / 2.0  # 1.46521
dt = 0.005
# dt = 0.0004  # stability?

joint_max_angle = 42.0 * np.pi / 180.0
# top_max_angle = 42.0 * np.pi / 180.0
# bot_max_angle = 42.0 * np.pi / 180.0
max_roll = 35.0 * np.pi / 180.0
max_pitch = 35.0 * np.pi / 180.0
max_yaw = 35.0 * np.pi / 180.0
max_leg_vel = 20.0 / 39.37
max_cart_table_acc = 8.0
max_cart_vel = 10.0  # human
max_cart_acc = 18.0  # human
max_angle_vel = 4.8  # human
max_angle_acc = 2100.0  # human

# warning: positive
# (think: to stay in place on earth, we are accelerating up wards to counteract
#   gravity)
gravity = np.array([0.0, 0.0, 9.81])
human_displacement = np.array([0.0, 0.0, 0.588])

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

# integration matrices for vestibular models
# for their meaning and derivation, see `vestibular.ipynb`
E0_acc = np.array([[0.98963583, -0.00129002], [0.00497405, 0.99999677]])
E1_acc = np.array([[4.97404728e-03], [1.24567102e-05]])
C_acc = np.array([[0.911, 0.0900068]])
E0_omega = np.array(
    [
        [9.50588885e-01, -6.42132787e-03, -1.59328193e-04],
        [4.87544271e-03, 9.99983811e-01, -4.01684389e-07],
        [1.22915423e-05, 4.99997291e-03, 9.99999999e-01],
    ]
)
E1_omega = np.array([[4.87544271e-03], [1.22915423e-05], [2.05721124e-08]])
C_omega = np.array([[10.09803922, 0.0, 0.0]])

# eigen-integration matrices for vestibular model
# cf. `vestibular.ipynb`
D_acc = np.array([0.99029738, 0.99933522])
P_acc_inv = np.array([[-1.20608765, -0.16040978], [-0.55520448, -1.082648]])
EP1_acc = np.array([[-0.00600114], [-0.0027751]])
CP_acc = np.array([[-0.76955197, 0.03088434]])

D_omega = np.array([0.95122942, 0.99950992, 0.99983335])
P_omega_inv = np.array(
    [
        [1.01838279e00, 1.33787544e-01, 3.32804834e-03],
        [-1.56830945e00, -1.57353713e01, -5.22769614e-01],
        [1.55148427e00, 1.56669489e01, 1.52106358e00],
    ],
)
EP1_omega = np.array([[0.00496671], [-0.00783963], [0.00775677]])
CP_omega = np.array([[10.04742719, 0.09659172, 0.01121379]])
