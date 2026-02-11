.PHONY: \
	all clean \
	mpc_example_clean pjrt_clean

all: mpc_example

clean: mpc_example_clean

###############
# mpc_example #
###############

mpc_example: src/mpc_example.cpp pjrt_exec mpc_export
	clang++ \
		-o mpc_example src/mpc_example.cpp artifacts/pjrt_exec.o \
		-std=c++17 -Wall -Wextra -O3 \
		-I./third_party/call_jax_from_cpp \
		-I. -I./third_party/highfive/include $(shell pkg-config --cflags hdf5) \
		$(shell pkg-config --libs hdf5) \
		-L./artifacts -Wl,-rpath,./artifacts -lpjrt_c_api_cpu_plugin_darwin

# -I. -I/opt/homebrew/Cellar/hdf5/1.14.6/include -I./third_party/highfive/include \
# -L/opt/homebrew/Cellar/hdf5/1.14.6/lib -lhdf5_cpp -lhdf5_hl -lhdf5 \

#		-L./artifacts -Wl,-rpath,./artifacts -lpjrt_c_api_cpu_plugin_linux

mpc_example_clean: pjrt_clean mpc_export_clean
	rm -f mpc_example

#############
# pjrt_exec #
#############

pjrt_exec: third_party/call_jax_from_cpp/Makefile
	mkdir -p artifacts && mkdir -p third_party/call_jax_from_cpp/artifacts
	cd third_party/call_jax_from_cpp/ && make pjrt_exec
	cp third_party/call_jax_from_cpp/artifacts/pjrt_exec.o artifacts/.

pjrt_clean:
	rm -f artifacts/pjrt_exec.o

##############
# mpc export #
##############

mpc_export: src/mpc_export.py
	python3 src/mpc_export.py

mpc_export_clean:
	rm -f artifacts/mpc_export.binpb artifacts/mpc_export.json
