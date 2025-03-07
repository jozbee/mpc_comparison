# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y \
  git
RUN git clone https://github.com/acados/acados.git ~/acados && \
  cd ~/acados && \
  git submodule update --init --recursive
RUN apt install -y \
  build-essential \
  cmake
RUN cd ~/acados && \
  mkdir -p build && \
  cd build && \
  cmake -DACADOS_WITH_QPOASES=ON .. && \
  make install -j
RUN apt install -y \
  python3 \
  python3-pip \
  python3-virtualenv
RUN mkdir ~/envs
RUN virtualenv ~/envs/acados_env --python=/usr/bin/python3
RUN echo "source /root/envs/acados_env/bin/activate" >> /root/.bashrc

SHELL ["/bin/bash", "-ic"]

RUN pip install -e /root/acados/interfaces/acados_template

RUN apt install -y \
  cargo
RUN git clone https://github.com/acados/tera_renderer.git ~/tera_renderer && \
  cd ~/tera_renderer && \
  cargo build --release
RUN cp /root/tera_renderer/target/release/t_renderer /root/acados/bin/t_renderer

ENV LD_LIBRARY_PATH="/root/acados/lib"
ENV ACADOS_SOURCE_DIR="/root/acados"
