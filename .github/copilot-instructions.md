# Copilot Instructions

## Project Overview

Experimental MPC (Model Predictive Control) implementations for a 6-DOF Stewart platform robot. Core algorithm: L-BFGS-B single-shooting MPC, written in JAX with a C++ integration layer via PJRT/XLA runtime.

## Commands

```bash
# Run all tests
pytest test/

# Run a single test file
pytest test/test_leg_spec.py

# Run a single test by name
pytest test/test_leg_spec.py::test_leg_spec

# Run visualization tests (interactive matplotlib)
pytest test/ --visualize

# Lint
ruff check .

# Type check
pyright

# Build C++ binary
cd cpp && make all

# Clean C++ build
cd cpp && make clean
```

## Architecture

The Python package is `exp_mpc.stewart_min` and is organized by concern:

- **`robo.py`** — `RoboParams` dataclass: Stewart platform geometry, leg limits, velocity/acceleration bounds.
- **`comp.py`** — JAX-based kinematics. `rot()` computes rotation matrices; `rot_dot()` / `rot_dot2()` compute first/second derivatives.
- **`opt.py`** — MPC optimizer. `Weights` dataclass holds cost function weights (acceleration, angular velocity, control, leg length, yaw). Wraps L-BFGS-B from scipy.
- **`utils.py`** — JAX utility functions: state propagation, array flattening/unflattening for optimizer interface.
- **`quartic_cost.py`** — Vestibular system cost via quartic polynomials.
- **`vest.py`** — `VSpec` dataclass: vestibular reference tracking specifications.
- **`viz.py`** / **`mp_mpl.py`** — Matplotlib visualization; `mp_mpl.py` handles multi-process rendering.

**C++ integration path:**
```
opt.py (JAX) → mpc_export.py (jax2exec) → mpc_export.binpb → mpc_example.cpp (PJRT runtime)
```

## Key Conventions

**JAX patterns:**
- 64-bit precision is always enabled: `jax.config.update("jax_enable_x64", True)`
- Use `jax.jit` with `static_argnames` for compile-time constants (e.g., array shapes, flags)
- Functional style — no mutation of JAX arrays in-place
- Prefer `jnp` over `np` for arrays that pass through jit boundaries

**Naming:**
- `_ref` suffix — reference/target trajectory (e.g., `accel_ref`)
- `_dot` suffix — time derivative (e.g., `rot_dot`)
- `_sim` suffix — simulated/predicted values
- `_irl` suffix — in-real-life / measured values

**Dataclasses:** Used extensively for parameter groups (`RoboParams`, `Weights`, `VSpec`). These are registered as JAX pytrees where needed for jit compatibility.

**Testing:** Tests that open matplotlib windows must be marked `@pytest.mark.visualize` and are skipped unless `--visualize` is passed.

**Line length:** 80 characters (Ruff).

**Data:** CSV files in `data/` are loaded with pandas then converted to JAX arrays. Do not commit large data files.
