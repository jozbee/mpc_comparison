#!/usr/bin/env bash

# should be called from `mpc_comparison/cpp`

if ! [[ "$(pip list)" == "exp_mpc" ]]; then
  pip install -e ..
fi
if ! [[ "$(pip list)" == "lbfgs" ]]; then
  pip install -e third_party/lbfgs
fi
if ! [[ "$(pip list)" == "" ]]; then
  pip install -e third_party/call_jax_from_cpp
fi
