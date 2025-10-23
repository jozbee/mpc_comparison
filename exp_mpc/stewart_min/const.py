import numpy as np


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
