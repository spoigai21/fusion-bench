"""Analytic traffic model: how many bytes each version *should* move.

Two jobs:

1. The prediction half of the argument. Phase 4's checkpoint is "you can state the
   expected byte ratio before looking at the counters" -- this module is that statement,
   written down before the profiler runs so it can be wrong in public.
2. Plan B. If Nsight counters are unavailable on the rented box, these numbers are what
   goes in the byte table, clearly labelled `derived` rather than `measured`.

All figures are for FP32 (4 bytes) and assume a perfect cache-free machine: every pass
over the row costs a full trip to DRAM. Real hardware has a 40 MB L2 on the A100, which
is exactly why the 4096x1024 shape (16 MB) is expected to break the model -- see
docs/notes.md.
"""

from __future__ import annotations

from dataclasses import dataclass

DTYPE_BYTES = 4


@dataclass(frozen=True)
class Traffic:
    reads: float  # passes over the N*D input
    writes: float  # passes over the N*D output
    note: str

    def bytes_for(self, n: int, d: int) -> int:
        return int((self.reads + self.writes) * n * d * DTYPE_BYTES)


# The ladder. Each rung removes one source of traffic and nothing else.
MODEL: dict[str, Traffic] = {
    "v0": Traffic(3.0, 1.0, "max pass + sum pass + normalize pass, all from global"),
    "v1": Traffic(1.0, 1.0, "row staged in shared memory once; both reductions local"),
    "v2": Traffic(1.0, 1.0, "single online pass; row held in registers for normalize"),
    "v2_generic": Traffic(2.0, 1.0, "fallback path: no register residency, re-reads row"),
    # Baselines, for context in the same units.
    "torch_softmax": Traffic(1.0, 1.0, "ATen persistent kernel: ideal traffic"),
    "torch_compile": Traffic(1.0, 1.0, "inductor fuses the composition; ideal if it works"),
    "naive_composition": Traffic(
        4.0, 2.0, "max -> sub/exp -> sum -> div as separate ops, materializing each"
    ),
    "numpy_f64": Traffic(float("nan"), float("nan"), "reference only, not timed on GPU"),
}

IDEAL = MODEL["v1"]  # 1 read + 1 write is the floor for any softmax


def predicted_bytes(kernel: str, n: int, d: int) -> int | None:
    t = MODEL.get(kernel)
    if t is None or t.reads != t.reads:  # NaN check
        return None
    return t.bytes_for(n, d)


def ideal_bytes(n: int, d: int) -> int:
    return IDEAL.bytes_for(n, d)


def predicted_ratio(a: str, b: str) -> float | None:
    """Predicted traffic of `a` divided by that of `b` -- the number the speedup should
    match if the kernel is bandwidth bound and the working set misses L2."""
    ta, tb = MODEL.get(a), MODEL.get(b)
    if not ta or not tb:
        return None
    return (ta.reads + ta.writes) / (tb.reads + tb.writes)


def gbps(total_bytes: float, micros: float) -> float:
    """Achieved bandwidth in GB/s from bytes moved and elapsed microseconds."""
    return total_bytes / (micros * 1e-6) / 1e9


if __name__ == "__main__":
    print("Predicted traffic (FP32, cache-free model)\n")
    print(f"{'kernel':<18}{'passes':>8}{'4096x1024':>14}{'4096x8192':>14}  note")
    for name, t in MODEL.items():
        if t.reads != t.reads:
            continue
        small = t.bytes_for(4096, 1024) / 1e6
        large = t.bytes_for(4096, 8192) / 1e6
        print(
            f"{name:<18}{t.reads + t.writes:>8.0f}{small:>12.0f} MB{large:>12.0f} MB  {t.note}"
        )
    print()
    print(f"predicted v0/v1 traffic ratio: {predicted_ratio('v0', 'v1'):.2f}x")
    print(f"predicted v0/v2 traffic ratio: {predicted_ratio('v0', 'v2'):.2f}x")
    print(f"predicted v1/v2 traffic ratio: {predicted_ratio('v1', 'v2'):.2f}x  (no traffic win;")
    print("    v2's win at large D is occupancy, not bytes -- see docs/notes.md)")
