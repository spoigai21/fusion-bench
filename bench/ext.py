"""JIT-compiles src/kernels.cu + src/bindings.cpp and exposes the module.

Phase 1 checkpoint:  python -c "import bench.ext"

Compiling on import (torch.utils.cpp_extension.load) rather than hand-rolling a build
keeps the architecture flags, include paths and the PyTorch ABI in one place. The first
import takes ~30-60 s; after that it is cached in .build/ and near-instant.

Environment overrides:
    FUSION_BENCH_VERBOSE=1   echo the nvcc command lines
    FUSION_BENCH_BUILD_DIR   where to cache objects (default: <repo>/.build)
"""

from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

_BUILD_DIR = Path(os.environ.get("FUSION_BENCH_BUILD_DIR", _ROOT / ".build"))
_BUILD_DIR.mkdir(parents=True, exist_ok=True)

CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    # -lineinfo costs nothing at runtime and lets ncu attribute counters back to source
    # lines, which is the difference between "v0 moves more bytes" and "*this* loop does".
    "-lineinfo",
]

ext = load(
    name="softmax_ext",
    sources=[str(_SRC / "bindings.cpp"), str(_SRC / "kernels.cu")],
    extra_include_paths=[str(_SRC)],
    extra_cflags=["-O3"],
    extra_cuda_cflags=CUDA_FLAGS,
    build_directory=str(_BUILD_DIR),
    verbose=bool(int(os.environ.get("FUSION_BENCH_VERBOSE", "0"))),
)

__all__ = ["ext", "CUDA_FLAGS"]


if __name__ == "__main__":
    import torch

    x = torch.randn(7, 13, device="cuda")
    y = ext.copy(x)
    torch.cuda.synchronize()
    assert torch.equal(x, y), "copy kernel did not round-trip"
    print("phase 1 ok: extension compiled, copy kernel round-trips")
    print(f"  build dir : {_BUILD_DIR}")
    print(f"  v2 path for D=1024 : {ext.v2_path(torch.empty(1, 1024, device='cuda'))}")
    print(f"  v2 path for D=8192 : {ext.v2_path(torch.empty(1, 8192, device='cuda'))}")
    print(f"  v1 smem for D=8192 : {ext.v1_smem_bytes(8192)} B")
