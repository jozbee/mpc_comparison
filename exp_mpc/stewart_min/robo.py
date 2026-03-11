"""Stewart platform specifications."""

from __future__ import annotations

import dataclasses

import numpy as np


# constants useful in the same context as robots
# (maybe should be specified somewhere else, but a special `const.py` file for
#  them seems a bit much)
gravity = np.array([0.0, 0.0, 9.81])
moon_gravity = np.array([0.0, 0.0, 1.625])


@dataclasses.dataclass
class RoboParams:
    """Robot parameters, mostly limits.

    All units are in SI.
    """

    dt: float = 0.005
    dt_mpc: float = 0.005 * 2.0

    leg_min: float = 1160.410000 * 1e-3 + 0.05
    leg_max: float = 1770.010000 * 1e-3 - 0.05
    leg_mid: float = (leg_min + leg_max) / 2.0
    joint_max_angle: float = np.deg2rad(42.0 - 5.0)

    max_euler: float = np.deg2rad(35.0)
    max_roll: float = max_euler
    max_pitch: float = max_euler
    max_yaw: float = max_euler
    max_rotary_yaw: float = np.deg2rad(90.0)

    max_leg_vel: float = 20.0 / 39.37
    max_rotary_vel: float = 0.5
    max_cart_table_acc: float = 8.0

    # human
    max_cart_vel: float = 10.0
    max_cart_acc: float = 18.0
    max_angle_vel: float = 4.8
    max_angle_acc: float = 2100.0

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


_human_displacement = np.array([0.0, 0.0, 0.588])
_bots = np.array(
    [
        [952.5055, 91.0723, -1410.0000],
        [-398.5396, 869.5826, -1409.8621],
        [-555.4801, 779.1038, -1410.0000],
        [-555.0219, -779.3507, -1409.6010],
        [-398.5396, -869.9006, -1410.0000],
        [952.7381, -89.7865, -1409.8718],
    ]
)
_bots *= 1e-3
_tops = np.array(
    [
        [314.3190, 327.5610, -215.5590],
        [126.7200, 435.8100, -215.4460],
        [-441.1510, 107.1130, -215.1930],
        [-441.5680, -109.0240, -215.6620],
        [126.7200, -436.9520, -215.6520],
        [314.7630, -328.8960, -215.7820],
    ]
)
_tops *= 1e-3

_bot_normals = np.array(
    [
        [-0.435014, 0.162005, 0.885729],
        [0.357803, -0.295803, 0.885708],
        [0.077200, -0.457799, 0.885698],
        [0.077200, 0.457799, 0.885698],
        [0.357803, 0.295803, 0.885708],
        [-0.435014, -0.162005, 0.885729],
    ]
)
_bot_normals /= np.linalg.norm(_bot_normals, axis=1)[:, np.newaxis]
_top_normals = np.array(
    [
        [0.435014, -0.162005, -0.885729],
        [-0.357803, 0.295803, -0.885708],
        [-0.077200, 0.457799, -0.885698],
        [-0.077200, -0.457799, -0.885698],
        [-0.357803, -0.295803, -0.885708],
        [0.435014, 0.162005, -0.885729],
    ]
)
_top_normals /= np.linalg.norm(_top_normals, axis=1)[:, np.newaxis]

_cart_home = np.array([0.0, 0.0, 0.1])  # home cartesian translation
_tops_home = np.array([top + _cart_home for top in _tops])
_lengths_home = float(np.mean(np.linalg.norm(_tops_home - _bots, axis=1)))


def _default_array(arr: np.ndarray):
    return dataclasses.field(default_factory=lambda: arr)


@dataclasses.dataclass
class RoboGeom:
    """Robot geometry.

    All units are in SI.

    Attributes
    ----------
    human_displacement :
        Cartesian translation vector from the robot frame to the human head
        frame.
    bots :
        6x3 array of the positions of the bottom joints in the robot frame.
    tops :
        6x3 array of the positions of the top joints in the robot frame.
    bot_normals :
        6x3 array of the normal vectors of the bottom joints.
    top_normals :
        6x3 array of the normal vectors of the top joints.
    cart_home :
        Cartesian translation vector to home.
    tops_home :
        6x3 array of the positions of the top joints in the home configuration.
    lengths_home :
        Scalar of the (average) leg lengths in the home configuration.
    """

    human_displacement: np.ndarray = _default_array(_human_displacement)
    bots: np.ndarray = _default_array(_bots)
    tops: np.ndarray = _default_array(_tops)
    bot_normals: np.ndarray = _default_array(_bot_normals)
    top_normals: np.ndarray = _default_array(_top_normals)
    cart_home: np.ndarray = _default_array(_cart_home)
    tops_home: np.ndarray = _default_array(_tops_home)
    lengths_home: float = _default_array(_lengths_home)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
