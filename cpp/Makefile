.PHONY: \
	all clean \
	mpc_example_clean pjrt_clean mpc_export_clean \
	pjrt_runtime

all: mpc_example

clean: mpc_example_clean

PIP?=pip
PYTHON?=python
PIP_LIST:= $(shell $(PIP) list)

###############
# mpc_example #
###############

mpc_example: src/mpc_example.cpp artifacts/pjrt_exec.o artifacts/mpc_export.binpb artifacts/mpc_export.json
	clang++ \
		-o mpc_example src/mpc_example.cpp artifacts/pjrt_exec.o \
		-std=c++17 -Wall -Wextra -O3 \
		-I./third_party/call_jax_from_cpp \
		-I. -I./third_party/highfive/include $(shell pkg-config --cflags hdf5) \
		$(shell pkg-config --libs hdf5) \
		-L./artifacts -Wl,-rpath,./artifacts -lpjrt_c_api_cpu_plugin

mpc_example_clean: pjrt_clean mpc_export_clean
	rm -f mpc_example

#############
# pjrt_exec #
#############

artifacts/pjrt_exec.o: third_party/call_jax_from_cpp/Makefile
	mkdir -p artifacts && mkdir -p third_party/call_jax_from_cpp/artifacts
	cd third_party/call_jax_from_cpp/ && make pjrt_exec
	cp third_party/call_jax_from_cpp/artifacts/pjrt_exec.o artifacts/.

pjrt_clean:
	rm -f artifacts/pjrt_exec.o

##############
# mpc export #
##############

artifacts/mpc_export.binpb artifacts/mpc_export.json: src/mpc_export.py
	@if ! [[ "$(PIP_LIST)" == *"exp_mpc"* ]]; then \
		$(PIP) install -e ..; \
	fi
	@if ! [[ "$(PIP_LIST)" == *"lbfgs"* ]]; then \
		$(PIP) install -e third_party/lbfgs; \
	fi
	@if ! [[ "$(PIP_LIST)" == *"jax2exec"* ]]; then \
		$(PIP) install -e third_party/call_jax_from_cpp; \
	fi
	 $(PYTHON) src/mpc_export.py

mpc_export_clean:
	rm -f artifacts/mpc_export.binpb artifacts/mpc_export.json

################
# pjrt runtime #
################

pjrt_runtime:
	./build_scripts/compile_xla_runtime.sh

pjrt_runtime_clean:
	./build_scripts/pjrt_cleanup.sh
