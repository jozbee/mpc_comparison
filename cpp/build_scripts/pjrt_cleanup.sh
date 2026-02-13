#!/usr/bin/env bash

if [[ "$(uname -s)" == "Darwin" ]]; then
  rm rm -f artifacts/libpjrt_c_api_cpu_plugin.dylib
elif [[ "$(uname -s)" == "Linux" ]]; then
  rm -f artifacts/libpjrt_c_api_cpu_plugin.so
fi
