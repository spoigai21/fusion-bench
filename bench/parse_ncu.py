#!/usr/bin/env python3
"""Turn raw ncu CSV into results/bytes.csv -- the three-row byte table that is the
evidence for the whole claim.

The README table has to be populatable without hand-typing a number, so this joins the
counters against results/summary.csv (for the median time) and computes achieved
bandwidth and % of peak here.

    python bench/parse_ncu.py results/raw/ncu_*.csv          # measured counters
    python bench/parse_ncu.py --analytic                     # Plan B, derived bytes

Plan B exists because many cloud containers refuse to expose hardware counters. If that
is the situation, the derived numbers still support the argument -- but every row is
labelled `derived` in the `source` column and the README must say so in words too.

Metric-name caution: Nsight metric names move between versions. Verify against
`ncu --query-metrics` on the actual machine rather than trusting a copied command.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import env  # noqa: E402
from bench.bytes_model import MODEL, gbps, ideal_bytes, predicted_bytes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Substring -> reported name. ncu demangles kernel names by default.
#
# Two mappings, because v2 shares its CUDA kernels with v2b and v2c: v2 dispatches to
# the vectorized kernel when it can and the generic one otherwise. A file profiled as
# the headline ladder reports those as "v2"; a file profiled as the attribution ladder
# reports them under their own names. The file's name selects the mapping, and
# profile_one.py refuses to launch both in one run, so a row is never ambiguous.
KERNEL_MAP = [
    ("softmax_v0_kernel", "v0"),
    ("softmax_v1_kernel", "v1"),
    ("softmax_v2_vec_kernel", "v2"),
    ("softmax_v2_gen_kernel", "v2"),
    # ATen's softmax has had several names across versions; match the common ones.
    ("softmax_v2a_kernel", "v2a"),
    ("cunn_SoftMaxForward", "torch_softmax"),
    ("softmax_warp_forward", "torch_softmax"),
    ("dispatch_softmax_forward", "torch_softmax"),
    ("SoftMaxForward", "torch_softmax"),
]

VARIANT_KERNEL_MAP = [
    ("softmax_v0_kernel", "v0"),
    ("softmax_v1_kernel", "v1"),
    ("softmax_v2a_kernel", "v2a"),
    ("softmax_v2_gen_kernel", "v2b"),
    ("softmax_v2_vec_kernel", "v2c"),
] + [m for m in KERNEL_MAP if m[1] == "torch_softmax"]

LADDER_ORDER = ("v0", "v1", "v2", "torch_softmax")
VARIANT_ORDER = ("v0", "v1", "v2a", "v2b", "v2c", "torch_softmax")

# ncu reports byte metrics with an auto-scaled unit unless --print-units base is passed.
# Handle both. Nsight uses SI prefixes for byte counters.
UNIT_SCALE = {
    "": 1.0, "byte": 1.0, "bytes": 1.0,
    "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
}

READ_METRICS = ("dram__bytes_read.sum",)
WRITE_METRICS = ("dram__bytes_write.sum",)
L2_METRICS = ("lts__t_bytes.sum",)


def classify(kernel_name: str, kernel_map=KERNEL_MAP) -> str | None:
    for needle, name in kernel_map:
        if needle in kernel_name:
            return name
    return None


def shape_from_filename(path: Path) -> tuple[int, int] | None:
    m = re.search(r"(\d+)x(\d+)", path.name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def read_ncu_csv(path: Path) -> list[dict]:
    """ncu prints banner lines before the CSV header; find the header and parse from it."""
    text = path.read_text(errors="replace").splitlines()
    start = None
    for i, line in enumerate(text):
        if '"ID"' in line and "Metric Name" in line:
            start = i
            break
    if start is None:
        raise SystemExit(
            f"{path}: no ncu CSV header found. Was the run a permissions failure?\n"
            f"  first lines: {text[:3]}"
        )
    return list(csv.DictReader(text[start:]))


def to_bytes(value: str, unit: str) -> float:
    v = float(value.replace(",", "").strip() or 0)
    return v * UNIT_SCALE.get(unit.strip(), 1.0)


def aggregate(rows: list[dict], kernel_map=KERNEL_MAP) -> dict[str, dict[str, float]]:
    """Sum each metric per ladder kernel. Several launches of the same kernel (e.g. the
    naive composition's elementwise ops) accumulate into one entry."""
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        name = classify(r.get("Kernel Name", ""), kernel_map)
        if name is None:
            continue
        metric = r.get("Metric Name", "").strip()
        b = to_bytes(r.get("Metric Value", "0"), r.get("Metric Unit", ""))
        slot = out.setdefault(name, {"read": 0.0, "write": 0.0, "l2": 0.0, "launches": 0.0})
        if metric in READ_METRICS:
            slot["read"] += b
            slot["launches"] += 1  # one row per (launch, metric)
        elif metric in WRITE_METRICS:
            slot["write"] += b
        elif metric in L2_METRICS:
            slot["l2"] += b
    return out


def load_timings(path: Path | None = None) -> dict[tuple[int, int, str], float]:
    """median_us keyed by (n, d, kernel), from results/summary.csv."""
    path = path or (RESULTS / "summary.csv")
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    out = {}
    for r in csv.DictReader(lines):
        if r.get("median_us"):
            out[(int(r["n"]), int(r["d"]), r["kernel"])] = float(r["median_us"])
    return out


FIELDS = [
    "n", "d", "kernel", "source", "launches",
    "dram_read_bytes", "dram_write_bytes", "dram_total_bytes", "l2_total_bytes",
    "predicted_bytes", "ideal_bytes", "dram_vs_ideal", "l2_vs_dram",
    "median_us", "dram_gbps", "pct_peak",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=[],
                    help="ncu --csv output files, named ncu_<N>x<D>.csv")
    ap.add_argument("--analytic", action="store_true",
                    help="Plan B: derive bytes from the model instead of counters")
    ap.add_argument("--variants", action="store_true",
                    help="with --analytic: emit v2a/v2b/v2c rows instead of v2")
    ap.add_argument("--shapes", default="4096x1024,4096x8192",
                    help="shapes to emit in --analytic mode")
    ap.add_argument("--summary", default=str(RESULTS / "summary.csv"),
                    help="timings to join against, for GB/s and % of peak")
    ap.add_argument("--out", default=str(RESULTS / "bytes.csv"))
    args = ap.parse_args()

    # Post-processing often happens off the GPU box; fall back to the env.json written
    # beside the timings so the % of peak column is not silently blank.
    info = env.merge_saved(env.collect(), Path(args.summary).parent / "env.json")
    peak = info.get("peak_bw_gbs")
    timings = load_timings(Path(args.summary))
    rows: list[dict] = []

    def emit(n: int, d: int, kernel: str, read: float, write: float, l2: float,
             launches: float, source: str) -> None:
        total = read + write
        ideal = ideal_bytes(n, d)
        pred = predicted_bytes(kernel, n, d)
        us = timings.get((n, d, kernel))
        row = {
            "n": n, "d": d, "kernel": kernel, "source": source,
            "launches": int(launches) if launches else "",
            "dram_read_bytes": int(read), "dram_write_bytes": int(write),
            "dram_total_bytes": int(total), "l2_total_bytes": int(l2),
            "predicted_bytes": pred if pred else "",
            "ideal_bytes": ideal,
            "dram_vs_ideal": f"{total / ideal:.2f}" if ideal else "",
            "l2_vs_dram": f"{l2 / total:.2f}" if total else "",
            "median_us": f"{us:.3f}" if us else "",
            "dram_gbps": "", "pct_peak": "",
        }
        if us and total:
            g = gbps(total, us)
            row["dram_gbps"] = f"{g:.1f}"
            if peak:
                row["pct_peak"] = f"{100 * g / peak:.1f}"
        rows.append(row)

    if args.analytic:
        print("Plan B: bytes DERIVED from the algorithm, not measured. Say so in the README.")
        for token in args.shapes.split(","):
            n, d = (int(v) for v in token.lower().strip().split("x"))
            for kernel in VARIANT_ORDER if args.variants else LADDER_ORDER:
                pred = predicted_bytes(kernel, n, d)
                if pred is None:
                    continue
                t = MODEL[kernel]
                read = t.reads * n * d * 4
                write = t.writes * n * d * 4
                emit(n, d, kernel, read, write, 0.0, 0, "derived")
    else:
        files = [Path(p) for pattern in (args.files or []) for p in sorted(glob.glob(pattern))]
        if not files:
            files = sorted((RESULTS / "raw").glob("ncu_*.csv"))
        if not files:
            raise SystemExit(
                "no ncu CSV files found. Run scripts/profile.sh first, "
                "or use --analytic for Plan B."
            )
        for path in files:
            shape = shape_from_filename(path)
            if not shape:
                print(f"skipping {path}: filename does not contain <N>x<D>")
                continue
            n, d = shape
            # "variants" in the filename selects the attribution mapping. scripts/
            # profile.sh writes ncu_variants_<N>x<D>.csv for those runs.
            variants = "variant" in path.name
            agg = aggregate(read_ncu_csv(path),
                            VARIANT_KERNEL_MAP if variants else KERNEL_MAP)
            if not agg:
                print(f"warning: {path} contained no recognized kernels")
            for kernel in (VARIANT_ORDER if variants else LADDER_ORDER):
                if kernel in agg:
                    a = agg[kernel]
                    emit(n, d, kernel, a["read"], a["write"], a["l2"], a["launches"], "measured")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        for line in env.header_lines(info):
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # Human-readable echo of the table that goes in the README.
    print(f"\n{'shape':<13}{'kernel':<15}{'DRAM MB':>10}{'L2 MB':>10}{'x ideal':>9}"
          f"{'us':>11}{'GB/s':>9}{'% peak':>8}   source")
    for r in rows:
        us = f"{float(r['median_us']):.1f}" if r["median_us"] else "-"
        print(f"{str(r['n']) + 'x' + str(r['d']):<13}{r['kernel']:<15}"
              f"{r['dram_total_bytes'] / 1e6:>10.1f}{r['l2_total_bytes'] / 1e6:>10.1f}"
              f"{r['dram_vs_ideal'] or '-':>9}{us:>11}{r['dram_gbps'] or '-':>9}"
              f"{r['pct_peak'] or '-':>8}   {r['source']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
