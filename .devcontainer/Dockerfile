FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt update -y && apt upgrade -y && apt install -y \
  build-essential \
  clang \
  git \
  python3 \
  python3-pip \
  python3-venv \
  npm \
  # need to also need runtime and headers for clang++, so we need g++-14
  g++-14 \
  # for building the PJRT runtime
  xxd \
  libhdf5-dev \
  pkg-config

RUN npm install -g @bazel/bazelisk

RUN mkdir ~/.virtualenvs
RUN python3 -m venv ~/.virtualenvs/jax
RUN echo "source ~/.virtualenvs/jax/bin/activate" >> ~/.bashrc
RUN source ~/.virtualenvs/jax/bin/activate && pip install \
  jax==0.9.0.1 \
  ipython
