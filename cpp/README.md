# Calling the MPC algorithm in C++

The JAX implementation of our MPC algorithm can be called from C++ with only a couple of dependencies.
The main technical details are implemented in [`call_jax_from_cpp`](https://github.com/jozbee/call_jax_from_cpp).
We have the following components:

* `mpc_export.py`: wraps the MPC algorithm, and uses the [`jax2exec`](https://github.com/jozbee/call_jax_from_cpp/tree/e1774cd614651b6e16b4ccff9f339243a881cf17/src/jax2exec) library to export the JAX function to a compiled binary.
This binary needs a run-time and an API to interface with, which is provided by `call_jax_from_cpp`.

* `mpc_example.cpp`: calls the binary from `mpc_export.py`.
After initializing the [PJRT runtime](https://openxla.org/xla/pjrt/pjrt_integration), calling the MPC algorithm amounts to transferring the inputs to the appropriate buffers and then calling a wrapper to access the binary.

* `Makefile`: specifies how to manually compile `mpc_example.cpp`.
Note that it calls the `Makefile` from `call_jax_from_cpp` in order to compile the PJRT wrappers.

Before calling `make all`, make sure that `libpjrt_c_api_cpu_plugin.so` (or `libpjrt_c_api_cpu_plugin.dylib` on Linux) is in the `cpp/artifacts/` directory.
There's a separate make command for this: `make pjrt_runtime`.
To run the program, call `make` in the `cpp` directory, and then call the built executable `mpc_example`.
To visualize the results, see `notebooks/cpp_analysis.ipynb`.

A Dockerfile is provided to demonstrate a minimal working environment, cf. `.devcontainer/Dockerfile`.
Assuming that you are in the root of the git repository, a typical initial usage follows:

```
$ cd .devcontainer
$ docker-compose build mpc_x86
$ docker-compose up -d mpc_x86
$ docker-compose exec -w /root/mpc_comparison mpc_x86 /bin/bash  # enter the container
# cd cpp
# make pjrt_runtime
# make  # compile mpc_example
# ./mpc_example
```

Note that on Linux, the default is to run docker commands as root.
However, with [some setup](https://docs.docker.com/engine/install/linux-postinstall/), you can run docker command as a non-root user.
