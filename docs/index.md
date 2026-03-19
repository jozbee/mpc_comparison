# Experimental MPC

Experimental MPC (Model Predictive Control) implementation for a Stewart
platform robot.

## Overview

`exp_mpc` implements a real-time nonlinear MPC controller for a Stewart
platform (hexapod) robot.
The controller is implemented in [JAX](https://jax.readthedocs.io/) to take advantage of the [XLA](https://github.com/openxla/xla) compiler and its [automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) capabilities.
Note that JAX provides a nice [numpy like](https://docs.jax.dev/en/latest/jax.numpy.html) interface, which eases the adoption process.
Because the code is compiled via XLA, it is possible to call the MPC code from C++.

## Installation

In the root git directory, run

```bash
pip install -e .
```

If you also want to build the docs, run

```bash
pip install -e ".[docs]"
```

## Docs

To build the docs, in the root git directory, run

```bash
sphinx-build -j auto -b html docs docs/_build/html
```

To access the docs locally, in your web browser, open
`docs/_build/html/index.html`.

## Package Layout

| Module | Description |
|---|---|
| {mod}`exp_mpc.stewart_min.robo` | Robot parameters and platform geometry |
| {mod}`exp_mpc.stewart_min.comp` | Kinematics, including derivatives |
| {mod}`exp_mpc.stewart_min.opt` | MPC cost spec and train step |
| {mod}`exp_mpc.stewart_min.utils` | Wrappers for MPC bookkeeping |
| {mod}`exp_mpc.stewart_min.vest` | Vestibular system spec |
| {mod}`exp_mpc.stewart_min.quartic_cost` | Quartic cost function (constraints) |
| {mod}`exp_mpc.stewart_min.viz` | Visualization routines |
| {mod}`exp_mpc.stewart_min.mp_mpl` | Multi-process visualization routines |

```{toctree}
:maxdepth: 2
:hidden:

cpp.md
api/index
```
