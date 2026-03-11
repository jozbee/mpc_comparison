# exp_mpc

Experimental MPC (Model Predictive Control) implementations for a 6-DOF Stewart
platform robot.

## Overview

`exp_mpc` implements a real-time-capable nonlinear MPC controller for a Stewart
platform (hexapod) robot. The core algorithm uses a single-shooting L-BFGS-B
optimization, implemented in [JAX](https://jax.readthedocs.io/) for
XLA/GPU-compatibility and automatic differentiation. A C++ integration layer
(via PJRT/XLA runtime) enables deployment in real-time control loops.

**Key capabilities:**

- Nonlinear MPC with L-BFGS-B optimizer (single-shooting formulation)
- JAX-based kinematics: rotation matrices, first and second derivatives
- Vestibular system cost modeling via quartic polynomial transfer functions
- Compiled C++ runtime via `jax2exec` + PJRT for real-time deployment
- Matplotlib-based visualization with multi-process rendering support

## Installation

```bash
pip install .
```

## Package Layout

The public API lives in `exp_mpc.stewart_min`:

| Module | Description |
|---|---|
| {mod}`exp_mpc.stewart_min.robo` | Robot parameters and platform geometry |
| {mod}`exp_mpc.stewart_min.comp` | JAX kinematics (rotations, derivatives) |
| {mod}`exp_mpc.stewart_min.opt` | MPC optimizer and cost weights |
| {mod}`exp_mpc.stewart_min.utils` | State propagation and array utilities |
| {mod}`exp_mpc.stewart_min.vest` | Vestibular system specifications |
| {mod}`exp_mpc.stewart_min.quartic_cost` | Quartic cost function |
| {mod}`exp_mpc.stewart_min.viz` | Visualization |
| {mod}`exp_mpc.stewart_min.mp_mpl` | Multi-process Matplotlib utilities |

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

api/index
```
