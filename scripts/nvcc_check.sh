#!/usr/bin/env bash
# Compile src/kernels.cu with the real nvcc, for the A100 target, using exactly the
# flags bench/ext.py passes. No GPU required -- nvcc needs a device to RUN a kernel,
# not to compile one. This meets the compile half of the phase 1 checkpoint on a
# laptop; the copy-kernel round-trip still needs silicon.
#
#   bash scripts/nvcc_check.sh              # sm_80 (A100)
#   ARCH=sm_89 bash scripts/nvcc_check.sh   # RTX 4090
#
# Only kernels.cu is compiled: it is deliberately torch-free, so it builds standalone.
# bindings.cpp needs libtorch headers and is not covered here.
#
# -Xptxas -v additionally reports registers per thread and shared memory per block for
# every kernel, which is the occupancy evidence behind the v1 vs v2 prediction in
# docs/notes.md -- obtainable here, without renting anything.
set -eu
REPO="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
IMAGE=nvidia/cuda:12.6.2-devel-ubuntu22.04
ARCH="${ARCH:-sm_80}"   # A100. sm_89 = RTX 4090, sm_90 = H100.

docker run --rm --platform linux/arm64 \
  -v "$REPO":/work -w /work "$IMAGE" \
  nvcc -arch="$ARCH" -O3 --use_fast_math -lineinfo \
       -Xptxas -v -Xcompiler -Wall \
       -Isrc -c src/kernels.cu -o /tmp/kernels.o
