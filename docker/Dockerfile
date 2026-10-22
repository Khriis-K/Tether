# syntax=docker/dockerfile:1
# The benchmark image (#19): Tether built on Linux on NVIDIA's TensorRT
# container, running the video-file benchmarks with one command. Build from the
# repo root with `docker build -f docker/Dockerfile -t tether .`; how to run it
# is in docs/DOCKER.md. TensorRT 11.3.0, CUDA 13.4.1, Ubuntu 24.04, Python 3.12.
FROM nvcr.io/nvidia/tensorrt:26.09-py3
LABEL org.opencontainers.image.source=https://github.com/Khriis-K/Tether

# GPU architectures nvcc compiles CUDA frame prep for: by default every one this
# base image supports (compute capability 7.5, Turing, to 12.0, Blackwell RTX 50xx),
# so the image runs on any of them. One is enough and builds faster, e.g. 120.
ARG CUDA_ARCHITECTURES="75;80;86;89;90;100;120"

# OpenCV reads video through FFmpeg's libraries. GLFW (the --show window) is
# always built, so it needs X11 and OpenGL headers even though Docker runs headless.
# libgl1 and libglib2.0-0 are for opencv-python. Ubuntu's docker-clean hook
# would empty the apt cache mount after every install, so it goes.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
      ninja-build pkg-config python3-venv \
      libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
      libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev \
      libgl1 libglib2.0-0

# CMake wants TensorRT as one folder with include/, lib/ and bin/; this image
# spreads it over the system folders.
RUN mkdir /opt/tensorrt-root \
    && ln -s /usr/include/x86_64-linux-gnu /opt/tensorrt-root/include \
    && ln -s /usr/lib/x86_64-linux-gnu /opt/tensorrt-root/lib \
    && ln -s /opt/tensorrt/bin /opt/tensorrt-root/bin

# The venv exists before CMake configures, so the tests' Python checks get its interpreter.
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv
WORKDIR /tether

# The C++ programs, with TensorRT and CUDA frame prep, before the Python
# package so a Python change doesn't rebuild OpenCV. Not the tests: building
# them runs trtexec, which needs the GPU that `docker build` doesn't have
# (run them with `docker run --gpus all tether cpp/build.sh`).
COPY cpp cpp
RUN cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTensorRT_ROOT=/opt/tensorrt-root \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DGLFW_BUILD_WAYLAND=OFF \
    && cmake --build cpp/build --target tether_pipeline_app tether_track tether_time_track

# Ultralytics (model export, the ByteTrack baseline) needs PyTorch; the CPU
# build is the one the dev laptop runs, and gigabytes smaller than the CUDA one.
# ONNX Runtime is pinned to the C++ pipeline's version, as on the laptop.
COPY pyproject.toml ./
COPY python python
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[test,demo,eval,bytetrack,export]" onnxruntime==1.31.0

COPY . .
# A Windows checkout (core.autocrlf) gives the scripts CRLF endings, which sh rejects.
RUN sed -i 's/\r$//' docker/*.sh cpp/build.sh
CMD ["sh", "docker/benchmark.sh"]
