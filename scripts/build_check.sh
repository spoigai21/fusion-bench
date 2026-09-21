#!/usr/bin/env bash
# Build the extension for real -- compile, link and import -- with no GPU.
#
# This runs the actual bench/ext.py load() path, not a hand-rolled compiler invocation:
# nvcc on kernels.cu, c++ on bindings.cpp, a link against libtorch_cuda/libc10_cuda/
# libcudart, and a Python import of the resulting .so. It meets the compile half of the
# phase 1 checkpoint. Only the kernel launch itself still needs silicon.
#
# It then asserts what can be asserted without a device: the pure-host entry points, and
# that every kernel rejects a CPU tensor. The dtype, dimension and contiguity checks sit
# behind the is_cuda check and cannot be reached here.
#
#   bash scripts/build_check.sh
#
# Requires Docker. First run pulls ~10 GB of CUDA image plus a CUDA torch wheel; after
# that it is cached. Optional -- the GPU box builds for real anyway.
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=fusion-bench-cuda
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # 8.0 = A100. 8.9 = RTX 4090, 9.0 = H100.

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE (one time, several GB)..."
  ctx=$(mktemp -d)
  cat > "$ctx/Dockerfile" <<EOF
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && apt-get -qq install -y --no-install-recommends \\
      python3-pip python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --upgrade pip
# The CUDA build, so libtorch_cuda.so and libc10_cuda.so exist to link against.
# Linking needs no GPU; only running the result does.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
RUN pip3 install --no-cache-dir ninja numpy
EOF
  docker build --platform linux/arm64 -t "$IMAGE" "$ctx"
  rm -rf "$ctx"
fi

# TORCH_CUDA_ARCH_LIST keeps cpp_extension from querying a device for its arch flags.
# FUSION_BENCH_BUILD_DIR keeps Linux objects out of the repo's .build/.
docker run --rm -i --platform linux/arm64 -v "$REPO":/work -w /work \
  -e TORCH_CUDA_ARCH_LIST="$ARCH_LIST" \
  -e FUSION_BENCH_BUILD_DIR=/tmp/build \
  "$IMAGE" python3 - <<'PY' 2>&1 | grep -vE "^==|CUDA Version|Container image|governed|By pulling|developer.nvidia|A copy of|WARNING: The NVIDIA|Use the NVIDIA|docs.nvidia|No CUDA runtime"
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
import bench.ext as E          # this is the compile + link + import
ext = E.ext
print("built and imported:", ext.__file__ if hasattr(ext, "__file__") else ext)

stats = {"ok": 0, "fail": 0}
def check(name, cond):
    print(("  [ok]   " if cond else "  [FAIL] ") + name)
    stats["ok" if cond else "fail"] += 1

print("\npure-host entry points:")
check("v1_smem_bytes(8192) == 33792", ext.v1_smem_bytes(8192) == 33792)
check("v1_smem_bytes(1024) == 5120", ext.v1_smem_bytes(1024) == 5120)

print("\nvalidation layer (fires before any launch):")
cpu = torch.randn(4, 8)
for fn in ("copy", "softmax_v0", "softmax_v1", "softmax_v2",
           "softmax_v2a", "softmax_v2b", "softmax_v2c"):
    try:
        getattr(ext, fn)(cpu)
        check(f"{fn} rejects a CPU tensor", False)
    except RuntimeError as e:
        check(f"{fn} rejects a CPU tensor", "must be a CUDA tensor" in str(e))

print(f"\n{stats['ok']} passed, {stats['fail']} failed")
print("NOT covered here: kernel launches, and the dtype/dim/contiguity checks that sit "
      "behind is_cuda. Both need a device.")
raise SystemExit(1 if stats["fail"] else 0)
PY
