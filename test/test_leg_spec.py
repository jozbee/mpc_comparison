import os
import shutil
import pytest
import numpy as np

import acados_leg.leg_spec as leg_spec
import test.helpers as helpers

@pytest.fixture
def leg_mpc():
    yield leg_spec.LegMPC()
    try:
        os.remove("acados_ocp.json")
    except FileNotFoundError:
        pass
    shutil.rmtree("c_generated_code", ignore_errors=True)

def test_leg_spec(leg_mpc: leg_spec.LegMPC):
    x0 = leg_spec.x_mid
    v0 = 0.01
    a_ref = 0.1
    x0_ref = leg_spec.x_mid
    leg_mpc.solve(x0, v0, x0_ref, a_ref)
    assert leg_mpc.solve_time < 1000.

@pytest.mark.visualize
def test_fixed_acc(leg_mpc: leg_spec.LegMPC):
    leg_mpc.reset()
    x0 = leg_spec.x_mid
    v0 = 0.01
    a_ref = 0.1
    x0_ref = leg_spec.x_mid
    
    x = []
    v = []
    a = []
    for _ in range(2**11):
        u = leg_mpc.solve(x0, v0, x0_ref, a_ref)
        print(f"time={leg_mpc.solve_time*1e6}us")

        x.append(x0)
        v.append(v0)
        a.append(u)

        x0 += u * leg_spec.dt**2 + v0 * leg_spec.dt
        v0 += u * leg_spec.dt

    a_ref = np.ones_like(a) * a_ref
    assert helpers.visualize_pass_fail(a_ref, a, "Fixed acceleration test (a)")

    x_ref = np.ones_like(x) * x0_ref
    assert helpers.visualize_pass_fail(x_ref, x, "Fixed acceleration test (x)")

    v_ref = np.zeros_like(v)
    assert helpers.visualize_pass_fail(v_ref, v, "Fixed acceleration test (v)")

def test_initial_impulse(leg_mpc: leg_spec.LegMPC):
    leg_mpc.reset()
    x0 = leg_spec.x_mid
    v0 = 0.01
    a_ref = 0.1
    x0_ref = leg_spec.x_mid

    x = []
    v = []
    a = []
    for _ in range(2**4):
        u = leg_mpc.solve(x0, v0, x0_ref, a_ref)
        print(f"time={leg_mpc.solve_time * 1e6}us")

        x.append(x0)
        v.append(v0)
        a.append(u)

        x0 += u * leg_spec.dt**2 + v0 * leg_spec.dt
        v0 += u * leg_spec.dt

    for _ in range(2**11):
        u = leg_mpc.solve(x0, v0, x0_ref, 0.0)
        print(f"time={leg_mpc.solve_time * 1e6}us")

        x.append(x0)
        v.append(v0)
        a.append(u)

        x0 += u * leg_spec.dt**2 + v0 * leg_spec.dt
        v0 += u * leg_spec.dt

    a_ref = np.ones_like(a) * a_ref
    assert helpers.visualize_pass_fail(a_ref, a, "Initial impulse test (a)")

    x_ref = np.ones_like(x) * x0_ref
    assert helpers.visualize_pass_fail(x_ref, x, "Initial impulse test (x)")

    v_ref = np.zeros_like(v)
    assert helpers.visualize_pass_fail(v_ref, v, "Initial impulse test (v)")
