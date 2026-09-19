#!/usr/bin/env bash
# Compile src/bindings.cpp against the REAL libtorch and CUDA headers, with no GPU.
#
# This is the companion to scripts/nvcc_check.sh: that one covers kernels.cu (which is
# torch-free by design), this one covers the binding layer. Together they mean the only
# unverified part of `make build` is the link and the actual kernel launch.
#
# A CPU-only torch wheel is used because it ships the c10/cuda headers. It omits one
# cmake-generated header (build-configuration defines, no API), which the image
# synthesizes -- so bindings.cpp is still checked against the genuine torch API.
#
#   bash scripts/torch_header_check.sh
#
# Requires Docker. First run pulls ~10 GB (the CUDA devel image) and a torch wheel;
# after that the image is cached and a check takes seconds. Optional: nothing in the
# project depends on it, and the GPU box compiles for real anyway.
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=fusion-bench-hdr
BASE=nvidia/cuda:12.6.2-devel-ubuntu22.04

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE (one time)..."
  ctx=$(mktemp -d)
  cat > "$ctx/Dockerfile" <<EOF
FROM $BASE
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && apt-get -qq install -y --no-install-recommends \\
      python3-pip python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN python3 -c "import os,pathlib,torch; p=pathlib.Path(os.path.dirname(torch.__file__))/'include/c10/cuda/impl'; p.mkdir(parents=True,exist_ok=True); (p/'cuda_cmake_macros.h').write_text('#pragma once\\n#define C10_CUDA_BUILD_SHARED_LIBS\\n')"
EOF
  docker build --platform linux/arm64 -t "$IMAGE" "$ctx"
  rm -rf "$ctx"
fi

# -std=c++20: torch >= 2.6 rejects anything older. The real build does not need this
# flag -- torch.utils.cpp_extension picks the standard itself -- but a hand-rolled g++
# invocation does.
docker run --rm --platform linux/arm64 -v "$REPO":/work -w /work "$IMAGE" bash -c '
set -e
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
python3 -c "import torch;print(\"libtorch\", torch.__version__)" 2>/dev/null
g++ -fsyntax-only -std=c++20 -Wall \
    -DTORCH_EXTENSION_NAME=softmax_ext \
    -I"$T/include" -I"$T/include/torch/csrc/api/include" \
    -I/usr/local/cuda/include \
    -I"$(python3 -c "import sysconfig;print(sysconfig.get_paths()[\"include\"])")" \
    -Isrc src/bindings.cpp
echo "bindings.cpp compiles against real libtorch headers"
' 2>&1 | grep -vE "^==|^$|CUDA Version|Container image|governed by|By pulling|developer.nvidia.com|A copy of this|WARNING: The NVIDIA|Use the NVIDIA|docs.nvidia.com|UserWarning|Triggered internally|_conversion_method"
