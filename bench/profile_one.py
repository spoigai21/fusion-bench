#!/usr/bin/env python3
"""One launch per kernel. This is the script ncu points at -- never the benchmark loop.

ncu instruments every launch it sees. Pointing it at 200 timed iterations gives 200 rows
and a run that takes minutes instead of seconds, so the byte table comes from here:
exactly one launch of each kernel, no warmup, nothing else on the GPU.

The input is built on the CPU and copied over, so tensor creation does not add a kernel
launch to the profile. Everything this script prints goes to stderr, because ncu writes
its CSV to stdout and anything else there would corrupt it.

Usage (under ncu -- see scripts/profile.sh):
    ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
        --print-units base --csv python bench/profile_one.py --shape 4096x8192
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from bench.baselines import naive_composition, torch_softmax  # noqa: E402
from bench.harness import make_input  # noqa: E402
from bench.kernels import KERNELS, v2_path  # noqa: E402


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="4096x8192", help="NxD")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernels", default="v0,v1,v2",
                    help="comma-separated subset of v0,v1,v2")
    ap.add_argument("--with-torch", action="store_true",
                    help="also launch torch.softmax (adds a row to the ncu output)")
    ap.add_argument("--with-naive", action="store_true",
                    help="also launch the naive composition (adds several rows)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log("CUDA not available")
        return 2

    n, d = (int(v) for v in args.shape.lower().split("x"))
    x = make_input(n, d, seed=args.seed)
    torch.cuda.synchronize()
    log(f"profiling {n}x{d} ({n * d * 4 / 1e6:.0f} MB input), v2 path {v2_path(x)}")

    # Exactly one launch each, in ladder order, so the ncu rows come back in a known
    # sequence even if the kernel names are mangled.
    for name in args.kernels.split(","):
        name = name.strip()
        if not name:
            continue
        log(f"  launching {name}")
        KERNELS[name](x)
        torch.cuda.synchronize()

    if args.with_torch:
        log("  launching torch.softmax")
        torch_softmax(x)
        torch.cuda.synchronize()

    if args.with_naive:
        log("  launching naive composition (several kernels)")
        naive_composition(x)
        torch.cuda.synchronize()

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
