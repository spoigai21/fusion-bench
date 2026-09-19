"""The ladder, in one place, so every script benchmarks the same three things."""

from __future__ import annotations

from bench.ext import ext

KERNELS = {
    "v0": ext.softmax_v0,  # naive: three global-memory passes
    "v1": ext.softmax_v1,  # fused: row staged in shared memory once
    "v2": ext.softmax_v2,  # online: single pass, warp shuffles, float4, registers
}

# The attribution ladder inside v2. Timed only with --variants, and kept out of KERNELS
# so the headline chart stays the three rungs the write-up is about.
#
# v2 itself dispatches to v2c where the shape allows and v2b otherwise, so these are not
# extra kernels so much as named entry points into paths that already exist -- except
# v2a, which is unique to the breakdown.
VARIANTS = {
    "v2a": ext.softmax_v2a,  # online pass only, shared-memory tree reduction
    "v2b": ext.softmax_v2b,  # + warp-shuffle reduction
    "v2c": ext.softmax_v2c,  # + float4 loads, row resident in registers
}

ALL_KERNELS = {**KERNELS, **VARIANTS}

DESCRIPTIONS = {
    "v0": "naive, 3 global passes",
    "v1": "fused via shared memory",
    "v2": "online + warp shuffle + float4",
    "v2a": "online pass only (tree reduction)",
    "v2b": "v2a + warp shuffle",
    "v2c": "v2b + float4 + registers",
}


def v2_path(x) -> str:
    """Which v2 code path this shape takes: vec<VPT> (register-resident) or generic."""
    return ext.v2_path(x)


def v1_smem_bytes(d: int) -> int:
    return int(ext.v1_smem_bytes(d))


__all__ = ["KERNELS", "VARIANTS", "ALL_KERNELS", "DESCRIPTIONS", "ext",
           "v2_path", "v1_smem_bytes"]
