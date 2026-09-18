"""The ladder, in one place, so every script benchmarks the same three things."""

from __future__ import annotations

from bench.ext import ext

KERNELS = {
    "v0": ext.softmax_v0,  # naive: three global-memory passes
    "v1": ext.softmax_v1,  # fused: row staged in shared memory once
    "v2": ext.softmax_v2,  # online: single pass, warp shuffles, float4, registers
}

DESCRIPTIONS = {
    "v0": "naive, 3 global passes",
    "v1": "fused via shared memory",
    "v2": "online + warp shuffle + float4",
}


def v2_path(x) -> str:
    """Which v2 code path this shape takes: vec<VPT> (register-resident) or generic."""
    return ext.v2_path(x)


def v1_smem_bytes(d: int) -> int:
    return int(ext.v1_smem_bytes(d))


__all__ = ["KERNELS", "DESCRIPTIONS", "ext", "v2_path", "v1_smem_bytes"]
