#!/usr/bin/env bash
# should be calling from mpc_comparison/cpp

cd third_party/call_jax_from_cpp/third_party/xla
./configure.py --backend=CPU --host_compiler=CLANG
bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.11 //xla/pjrt/c:pjrt_c_api_cpu_plugin.so

cd ../../../..
mkdir -p artifacts
cp third_party/call_jax_from_cpp/third_party/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so artifacts/libpjrt_c_api_cpu_plugin.so
if [[ "$OS_NAME" == "Darwin" ]]; then
  cd artifacts
  mv libpjrt_c_api_cpu_plugin.so libpjrt_c_api_cpu_plugin.dylib
  install_name_tool -id "$(pwd)/libpjrt_c_api_cpu_plugin.dylib" libpjrt_c_api_cpu_plugin.dylib
  # echo "$(otool -L libpjrt_c_api_cpu_plugin.dylib)"
  cd ..
fi
