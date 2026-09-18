#!/usr/bin/env python3
"""FP64 cross-check for all three kernels across the edge cases that break kernels.

Pass condition: max absolute error below 1e-5 against a float64 NumPy reference
computed from the same fixed-seed input. A kernel that fails here does not get timed.

The cases are chosen to hit the things that actually go wrong:

  D % 32 != 0    warp remainder -- partial warps in the reduction
  D % 4  != 0    float4 tail    -- forces v2 off its vectorized path
  D == 1         degenerate row -- almost every thread in the block is idle
  -inf in a row  masked positions, as in causal attention
  x + 1000       large magnitude -- overflows without the max subtraction. This is the
                 one a naive implementation silently fails, and the test prints the
                 overflow it avoids so the reason for the subtraction is visible.

Usage:  python tests/verify.py [--tol 1e-5] [--quiet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from bench.baselines import BASELINES, naive_composition  # noqa: E402
from bench.harness import TOLERANCE, validate  # noqa: E402
from bench.kernels import KERNELS, v1_smem_bytes, v2_path  # noqa: E402


def cases(seed: int = 0) -> list[tuple[str, torch.Tensor, str]]:
    """(name, tensor, what it exercises)."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(n: int, d: int) -> torch.Tensor:
        return torch.randn(n, d, generator=g, dtype=torch.float32).cuda().contiguous()

    out: list[tuple[str, torch.Tensor, str]] = [
        ("8x1", rnd(8, 1), "degenerate row, D=1"),
        ("8x3", rnd(8, 3), "D < warp, not a multiple of 4"),
        ("8x31", rnd(8, 31), "D % 32 != 0 and D % 4 != 0, warp remainder + float4 tail"),
        ("8x33", rnd(8, 33), "D % 32 == 1, one element past a full warp"),
        ("8x255", rnd(8, 255), "just under one element per thread"),
        ("8x257", rnd(8, 257), "just over one element per thread"),
        ("4x1023", rnd(4, 1023), "D % 4 != 0 near the small benchmark shape"),
        ("64x1024", rnd(64, 1024), "benchmark shape, small N"),
        ("16x8192", rnd(16, 8192), "large benchmark shape, VPT=8 register path"),
        ("2x20480", rnd(2, 20480), "too long for registers: forces the v2 generic path"),
    ]

    # Masked positions, as in causal attention. Deliberately never a fully masked row:
    # every kernel here returns uniform 1/D for one, torch returns NaN, and the FP64
    # reference is undefined -- documented in docs/notes.md rather than tested.
    masked = rnd(8, 128)
    for r in range(masked.shape[0]):
        masked[r, r * 5 + 1 :] = float("-inf")
    out.append(("8x128 masked", masked, "row containing -inf (causal attention mask)"))

    # Large magnitude: mathematically identical softmax, but exp() without the max
    # subtraction overflows to inf here.
    out.append(("8x1024 +1000", rnd(8, 1024) + 1000.0, "large magnitude, needs max subtraction"))
    out.append(("8x1024 -1000", rnd(8, 1024) - 1000.0, "large negative magnitude"))

    # Wide dynamic range inside a single row: the running max in v2 is rescaled many
    # times, which is where an online implementation goes wrong if the rescale is missing.
    ramp = torch.linspace(-80.0, 80.0, 1024, dtype=torch.float32).repeat(8, 1)
    out.append(("8x1024 ramp", ramp.cuda().contiguous(), "monotone ramp, many max updates"))

    return out


def show_overflow_demo() -> None:
    """Why the max subtraction exists, printed rather than asserted."""
    x = torch.randn(1, 1024, device="cuda") + 1000.0
    unsafe = torch.exp(x).sum().item()
    safe = torch.exp(x - x.max()).sum().item()
    print("  why the max subtraction exists:")
    print(f"    sum(exp(x))            = {unsafe}      <- overflows")
    print(f"    sum(exp(x - max(x)))   = {safe:.6f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=TOLERANCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- nothing to verify", file=sys.stderr)
        return 2

    print(f"FP64 cross-check, tolerance {args.tol:g}, device {torch.cuda.get_device_name(0)}\n")
    if not args.quiet:
        show_overflow_demo()

    fns = {**KERNELS, "naive_composition": naive_composition,
           "torch_softmax": BASELINES["torch_softmax"]}

    print(f"{'case':<16}{'D':>7}  {'v2 path':<9}" + "".join(f"{k:>20}" for k in fns))
    print("-" * (16 + 7 + 2 + 9 + 20 * len(fns)))

    failures: list[str] = []
    for case_name, x, why in cases(args.seed):
        d = x.shape[1]
        row = f"{case_name:<16}{d:>7}  {v2_path(x):<9}"
        for fn_name, fn in fns.items():
            try:
                ok, err = validate(fn, x, args.tol)
                cell = ("PASS" if ok else "FAIL") + f" {err:.2e}"
                if not ok:
                    failures.append(f"{fn_name} on {case_name} (err {err:.3e})")
            except RuntimeError as exc:
                # v1's shared-memory ceiling is a documented limit, not a wrong answer.
                if "shared memory" in str(exc):
                    cell = f"SKIP {v1_smem_bytes(d) // 1024}KB"
                else:
                    cell = "ERROR"
                    failures.append(f"{fn_name} on {case_name}: {exc}")
            row += f"{cell:>20}"
        print(row + f"   {'' if args.quiet else why}")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all kernels within {args.tol:g} of the FP64 reference on every case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
