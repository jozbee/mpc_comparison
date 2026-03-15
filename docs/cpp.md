# C++ interface

Because the MPC code can be (JIT) compiled with JAX, it is possible to interface with the compiled binary from C++.
The main python to C++ pipeline is developed in [call_jax_from_cpp](https://github.com/jozbee/call_jax_from_cpp).
The build dependencies are minimal: a few source files and a shared library.

## Building and running the example

First, make sure the submodules are initialized.
If not, run `git submodule update --init --recursive`.
Then just call `make` in the `cpp` directory.
The [`Makefile`](https://github.com/jozbee/mpc_comparison/blob/main/cpp/Makefile) should handle all the python dependencies.
The other build dependencies are specified in a [`Dockerfile`](https://github.com/jozbee/mpc_comparison/blob/main/.devcontainer/Dockerfile).
Namely, we need all the usual development tools, along with [bazel](https://github.com/bazelbuild/bazel), python, and a specific version of JAX.
An example usage of using the docker container follows.

```
$ cd .devcontainer
$ docker-compose build mpc_x86
...
$ docker-compose up -d mpc_x86
...
$ docker-compose exec -w /root/mpc_comparison mpc_x86 /bin/bash
(jax) root@docker-desktop:~/mpc_comparison# cd cpp
(jax) root@docker-desktop:~/mpc_comparison# make
...
(jax) root@docker-desktop:~/mpc_comparison# ./mpc_example
Average timing: 4926.28 microseconds
Stddev timing: 587.665 microseconds
Min timing: 4683 microseconds
Max timing: 6390 microseconds
Min timing index: 1984
Max timing index: 1357
Output data: -0.0246488, 0.000437004
```

(The example was run on Apple Silicon, which is emulating the x86 architecture.
You should expect better performance on bare metal.)

## Integration notes

The Python to C++ pipeline consists of API wrappers and some minimal dependencies.
First, we need python code to tell the C++ code what the input/output should be.
In [`cpp/src/mpc_export.py`](https://github.com/jozbee/mpc_comparison/blob/main/cpp/src/mpc_export.py), this is specified by `mpc_solver`.
However, most of the inputs to `mpc_solver` are not accessible in the final executable.
Most of the parameters are statically compiled.
So, tuning is done in the python file, independently of compiling any C++ code.
See [Tuning](#tuning) for tuning notes.

After the python code is specified, we use the library [`jax2exec`](https://github.com/jozbee/call_jax_from_cpp/tree/main/src/jax2exec) to give us access to the compiled binary for our MPC algorithm.
This binary is dynamically loaded by the [`pjrt_exec`](https://github.com/jozbee/call_jax_from_cpp/tree/main/src/pjrt_exec) interface to run our MPC binary.
See [`call_jax_from_cpp`](https://github.com/jozbee/call_jax_from_cpp/tree/main?tab=readme-ov-file) for more information.

The `call_jax_from_cpp` dependencies are pretty minimal.
See the provided [`Makefile`](https://github.com/jozbee/mpc_comparison/blob/main/cpp/Makefile) for the specific build process.
Essentially, we just need a few header files, from [`pjrt_exec`](https://github.com/jozbee/call_jax_from_cpp/tree/main/src/pjrt_exec) and [`OpenXLA`](https://github.com/openxla/xla/tree/main/xla/pjrt/c).
Note that our `Makefile` also makes use of [HDF5](https://github.com/HDFGroup/hdf5), but this is only used to save the MPC data for visualization in Python.
HDF5 is not needed to use the `call_jax_from_cpp` code.

## Tuning

Tuning is performed after the `if __name__ == "__main__":` block in `mpc_export.py`.
After tuning, we do not need to recompile any C++ code.
The compiled JAX binary is dynamically loaded by `call_jax_from_cpp`, so we just need to put the artifacts where the C++ code can find them.
See our example code [`mpc_example.cpp`](https://github.com/jozbee/mpc_comparison/blob/main/cpp/src/mpc_example.cpp).

The tuning parameters are the first group of inputs to `mpc_solver`.
We provide a table.

| Parameter | Description |
| --- | --- |
| `max_iter` | Maximum number of L-BFGS solver iterations. |
| `max_ls` | Maximum number of L-BFGS line search iterations.|
| `tol` | Early-termination tolerance for L-BFGS. |
| `c1` | [Wolfe condition](https://en.wikipedia.org/wiki/Wolfe_conditions) constant for L-BFGS line search. Should probably leave at `1e-4`. |
| `c2` | Another Wolfe condition constant for L-BFGS. Should probably leave at `0.9`. |
| `n` | Number of cycles in MPC horizon. |
| `robo_params` | Robot parameters as an instance of {py:class}`exp_mpc.stewart_min.robo.RoboParams`. See the docs. |
| `robo_geom` | Robot geometry as an instance of {py:class}`exp_mpc.stewart_min.robo.RoboGeom`. See the docs. |
| `vspec_acc` | Vestibular model for linear acceleration. An instance of {py:class}`exp_mpc.stewart_min.vest.VSpec`. See the docs. |
| `vspec_omega` | Vestibular model for angular velocity. An instance {py:class}`exp_mpc.stewart_min.vest.VSpec`. See the docs.|
| `vspec_acc_mpc` | MPC vestibular model for linear acceleration. The MPC algorithm can use a different control cycle time than the real robot, so many of the attributes differ numerically with `vspec_acc`. Note that `vspec_acc_mpc` should use the same transfer function as `vspec_acc`. An instance {py:class}`exp_mpc.stewart_min.vest.VSpec`. See the docs. |
| `vspec_omega_mpc` | MPC vestibular model for angular velocity. The MPC algorithm can use a different control cycle time than the real robot, so many of the attributes differ numerically with `vspec_omega`. Note that `vspec_omega_mpc` should use the same transfer function as `vspec_omega`. An instance {py:class}`exp_mpc.stewart_min.vest.VSpec`. See the docs. |
| `weights` | Weight scaling (including expoential-time scaling) for cost function. An instance of {py:class}`exp_mpc.stewart_min.opt.ExpWeights`. See the docs. |
| `cost_terms` | Soft constraint functions for robot. An instance of {py:class}`exp_mpc.stewart_min.opt.CostTerms`. See the docs. |
| `use_terminal` | True to include a terminal cost, and False otherwise. A terminal cost can improve tracking back to home while not destroying dynamic performance, if tuned properly. |
