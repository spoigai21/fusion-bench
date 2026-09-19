#!/usr/bin/env python3
"""Full run: validate, time, and write results/summary.csv (+ results/raw/*.csv).

Two shapes only, per the plan:
    4096x1024  -- 16 MB working set, fits inside the A100's 40 MB L2. The DRAM model is
                  expected to break here, and that is the interesting result.
    4096x8192  -- 128 MB, well past L2. The model should hold.

Nothing is timed until it has passed the FP64 cross-check at that exact shape.

Usage:
    python bench/run_all.py                          # both shapes, 50 warmup + 200 timed
    python bench/run_all.py --shapes 4096x1024
    python bench/run_all.py --iters 50 --warmup 10   # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from bench import env  # noqa: E402
from bench.baselines import BASELINES, reset_compile_cache  # noqa: E402
from bench.bytes_model import gbps, ideal_bytes, predicted_bytes  # noqa: E402
from bench.harness import ITERS, TOLERANCE, WARMUP, make_input, time_fn, validate, write_raw  # noqa: E402
from bench.kernels import (  # noqa: E402
    DESCRIPTIONS, KERNELS, VARIANTS, v1_smem_bytes, v2_path,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

FIELDS = [
    "n", "d", "kernel", "kind", "status", "max_abs_err",
    "median_us", "p5_us", "p95_us", "mean_us", "stdev_us",
    "model_bytes", "model_gbps", "effective_gbps", "effective_pct_peak",
    "speedup_vs_v0", "speedup_vs_torch",
    "v2_path", "v1_smem_bytes", "warmup", "iters", "sm_clock_mhz",
]


def parse_shapes(spec: str) -> list[tuple[int, int]]:
    shapes = []
    for token in spec.split(","):
        n, d = token.lower().strip().split("x")
        shapes.append((int(n), int(d)))
    return shapes


def current_sm_clock() -> str:
    return env.nvidia_smi_clocks().get("sm_clock", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="4096x1024,4096x8192")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--tol", type=float, default=TOLERANCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=str(RESULTS),
                    help="where summary.csv, env.json and raw/ are written")
    ap.add_argument("--out", default=None,
                    help="override the summary.csv path (default: <results-dir>/summary.csv)")
    ap.add_argument("--variants", action="store_true",
                    help="also time v2a/v2b/v2c, the attribution ladder inside v2")
    ap.add_argument("--skip-compile", action="store_true",
                    help="skip the torch.compile baseline (inductor can be slow to build)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    # Everything this run produces goes under one directory, so a run pointed somewhere
    # else cannot leave files scattered in the repo's results/.
    results_dir = Path(args.results_dir)
    out = Path(args.out) if args.out else results_dir / "summary.csv"

    info = env.collect()
    env.write_env_json(results_dir / "env.json")
    peak = info.get("peak_bw_gbs")

    print(f"GPU        : {info.get('gpu')}  ({info.get('sm_count')} SMs)")
    print(f"peak BW    : {peak} GB/s  [{info.get('peak_bw_source')}]")
    print(f"torch/cuda : {info.get('torch')} / {info.get('cuda_runtime')}")
    print(f"clocks     : sm {info.get('clocks_sm_clock', 'n/a')} "
          f"(max {info.get('clocks_sm_clock_max', 'n/a')}), "
          f"mem {info.get('clocks_mem_clock', 'n/a')} "
          f"(max {info.get('clocks_mem_clock_max', 'n/a')})")
    print(f"throttle   : {info.get('clocks_throttle_reasons', 'n/a')}")
    print(f"protocol   : {args.warmup} warmup + {args.iters} timed, CUDA events, median\n")

    fns = dict(KERNELS)
    if args.variants:
        fns.update(VARIANTS)
    for name, fn in BASELINES.items():
        if args.skip_compile and name == "torch_compile":
            continue
        fns[name] = fn

    rows: list[dict] = []
    header = env.header_lines(info)

    for n, d in parse_shapes(args.shapes):
        x = make_input(n, d, seed=args.seed)
        path = v2_path(x)
        smem = v1_smem_bytes(d)
        working_mb = n * d * 4 / 1e6
        print(f"== {n}x{d}  ({working_mb:.1f} MB input, v2 path {path}, "
              f"v1 shared mem {smem / 1024:.1f} KB/block)")

        reset_compile_cache()  # inductor specializes per shape; never recompile while timing

        shape_rows: dict[str, dict] = {}
        for name, fn in fns.items():
            kind = "kernel" if name in KERNELS else ("variant" if name in VARIANTS
                                                     else "baseline")
            row = {f: "" for f in FIELDS}
            row.update({
                "n": n, "d": d, "kernel": name, "kind": kind,
                "warmup": args.warmup, "iters": args.iters,
                "v2_path": path if name == "v2" else "",
                "v1_smem_bytes": smem if name == "v1" else "",
            })

            try:
                ok, err = validate(fn, x, args.tol)
            except RuntimeError as exc:
                # A limit the kernel states up front (v1's shared-memory ceiling, v2c's
                # vectorization requirement) is a skip, not a failure. The kernels own
                # those rules; nothing here re-derives them.
                msg = str(exc).splitlines()[0]
                limit = "documented limit" in msg
                row["status"] = "skip" if limit else "error"
                print(f"  {name:<20} {'skip ' if limit else 'ERROR'}   {msg[:90]}")
                rows.append(row)
                continue

            row["max_abs_err"] = f"{err:.3e}"
            if not ok:
                # A kernel that fails correctness does not get timed. Reporting a time
                # for a wrong answer is the one thing that would invalidate the project.
                row["status"] = "FAIL"
                print(f"  {name:<20} FAIL    max abs err {err:.3e} > {args.tol:g}  (not timed)")
                rows.append(row)
                continue

            t = time_fn(fn, x, warmup=args.warmup, iters=args.iters)
            model_b = predicted_bytes("v2_generic" if (name == "v2" and path == "generic")
                                      else name, n, d)
            row.update({
                "status": "pass",
                "median_us": f"{t.median_us:.3f}",
                "p5_us": f"{t.p5_us:.3f}",
                "p95_us": f"{t.p95_us:.3f}",
                "mean_us": f"{t.mean_us:.3f}",
                "stdev_us": f"{t.stdev_us:.3f}",
                "sm_clock_mhz": current_sm_clock(),
            })
            if model_b:
                # model_gbps counts the bytes this version actually moves, so a wasteful
                # kernel can score high on it. effective_gbps counts only the useful
                # bytes (one read + one write), which is the figure that is comparable
                # across the ladder and the one quoted as "% of peak".
                row["model_bytes"] = model_b
                row["model_gbps"] = f"{gbps(model_b, t.median_us):.1f}"

            ideal_g = gbps(ideal_bytes(n, d), t.median_us)
            row["effective_gbps"] = f"{ideal_g:.1f}"
            if peak:
                row["effective_pct_peak"] = f"{100.0 * ideal_g / peak:.1f}"

            print(f"  {name:<20} {t.median_us:9.2f} us  "
                  f"[p5 {t.p5_us:7.2f}  p95 {t.p95_us:7.2f}]  "
                  f"err {err:.1e}  "
                  f"{ideal_g:7.0f} GB/s effective"
                  + (f" ({100 * ideal_g / peak:.0f}% of peak)" if peak else ""))

            write_raw(results_dir / "raw" / f"{name}_{n}x{d}.csv", t, header)
            shape_rows[name] = row
            rows.append(row)

        # Ratios are what the write-up actually claims, so they are computed here rather
        # than typed into the README by hand.
        base = shape_rows.get("v0", {}).get("median_us")
        tsm = shape_rows.get("torch_softmax", {}).get("median_us")
        for name, row in shape_rows.items():
            me = float(row["median_us"])
            if base:
                row["speedup_vs_v0"] = f"{float(base) / me:.3f}"
            if tsm:
                row["speedup_vs_torch"] = f"{float(tsm) / me:.3f}"
        print()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        for line in header:
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}")
    print(f"      {results_dir / 'raw'}/  ({len([r for r in rows if r['status'] == 'pass'])} "
          f"timing files, {args.iters} samples each)")
    print(f"      {results_dir / 'env.json'}")

    failed = [r for r in rows if r["status"] in ("FAIL", "error")]
    if failed:
        print(f"\n{len(failed)} entr(ies) did not pass and were not timed:")
        for r in failed:
            print(f"  - {r['kernel']} at {r['n']}x{r['d']}: {r['status']}")
        return 1

    print("\nkernel descriptions: " + ", ".join(f"{k}={v}" for k, v in DESCRIPTIONS.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
