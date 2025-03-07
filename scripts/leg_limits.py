"""Test the limits of the prismatic joint."""

v_max = 0.1
a_max = 0.2
dt = 1. / 250.

x0 = 1.5
v0 = v_max

x1 = v0 * dt + x0

xs = []
while x1 != x0:
    xs.append(x1)
    v0 =  (x1 - x0) / dt
    x0 = x1

    x_acc_pos = a_max * dt**2 + v0 * dt + x0
    x_acc_neg = -a_max * dt**2 + v0 * dt + x0
    x_vel_pos = v_max * dt + x0
    x_vel_neg = -v_max * dt + x0

    x1 = x0
    x1 = max(x1, x_acc_neg, x_vel_neg)
    x1 = min(x1, x_acc_pos, x_vel_pos)
