#!/usr/bin/env python3
"""Regenerate the README tables from results/. No number is ever typed by hand.

Phase 6's checkpoint is that the write-up can be populated from summary.csv and
bytes.csv mechanically. This is that step: it rewrites the regions of README.md
delimited by <!-- BEGIN:name --> / <!-- END:name --> markers and leaves the prose alone.

    python bench/make_tables.py            # rewrite README.md in place
    python bench/make_tables.py --check    # exit 1 if it would change anything (CI)
    python bench/make_tables.py --stdout   # print the tables, touch nothing

Missing inputs are not an error: a table with no data keeps its em-dash placeholders,
so this is safe to run before the GPU session as well as after.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

LADDER = ["v0", "v1", "v2"]
REFS = ["torch_softmax", "torch_compile"]
VARIANTS = ["v2a", "v2b", "v2c"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def shapes_of(rows: list[dict]) -> list[tuple[int, int]]:
    return sorted({(int(r["n"]), int(r["d"])) for r in rows})


def cell(value: str | None, suffix: str = "") -> str:
    return f"{value}{suffix}" if value else "—"


def timings_table(rows: list[dict]) -> str:
    cols = LADDER + REFS
    head = "| shape | " + " | ".join(
        f"`{c}`" if c.startswith("torch") else c for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    if not rows:
        body = [f"| 4096×{d} |" + " — |" * len(cols) for d in (1024, 8192)]
        return "\n".join([head, sep, *body])

    out = [head, sep]
    for n, d in shapes_of(rows):
        by = {r["kernel"]: r for r in rows if int(r["n"]) == n and int(r["d"]) == d}
        cells = []
        for c in cols:
            us = by.get(c, {}).get("median_us")
            cells.append(f"{float(us):,.0f} µs" if us else "—")
        out.append(f"| {n}×{d} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def headline_table(rows: list[dict], n: int, d: int) -> str:
    """The ladder as a progression: each rung against the one below it.

    Rung, absolute figure, speedup over the previous rung, and position against the
    reference implementation.
    """
    out = ["| Rung | median | effective GB/s | % of HBM peak | Speedup vs prev | vs `torch.softmax` |",
           "|---|---|---|---|---|---|"]
    by = {r["kernel"]: r for r in rows if int(r["n"]) == n and int(r["d"]) == d}
    if not by:
        return "\n".join(out + ["| v0 | — | — | — | — | — |"])

    tsm = by.get("torch_softmax", {}).get("median_us")
    prev = None
    order = [("v0", "`softmax_v0` naive"), ("v1", "`softmax_v1` fused"),
             ("v2", "`softmax_v2` online"), ("torch_softmax", "`torch.softmax` (ATen)"),
             ("torch_compile", "`torch.compile`"),
             ("naive_composition", "naive composition (multi-kernel)")]
    for key, label in order:
        r = by.get(key)
        if not r or not r.get("median_us"):
            continue
        us = float(r["median_us"])
        # The reference rows are not rungs, so they get no "vs prev" figure.
        is_rung = key in ("v0", "v1", "v2")
        step = f"{prev / us:.2f}×" if (is_rung and prev) else "—"
        vs_t = f"{float(tsm) / us:.2f}×" if tsm else "—"
        bold = "**" if key == "v2" else ""
        out.append(
            f"| {label} | {bold}{us:,.1f} µs{bold} | {cell(r.get('effective_gbps'))} "
            f"| {cell(r.get('effective_pct_peak'), '%')} | {bold}{step}{bold} | {bold}{vs_t}{bold} |")
        if is_rung:
            prev = us
    return "\n".join(out)


def speedup_table(rows: list[dict]) -> str:
    """Every ratio the write-up claims, computed rather than asserted."""
    out = ["| shape | v0 → v1 | v1 → v2 | v0 → v2 | v2 vs `torch.softmax` |",
           "|---|---|---|---|---|"]
    if not rows:
        return "\n".join(out + ["| 4096×1024 | — | — | — | — |",
                                "| 4096×8192 | — | — | — | — |"])
    for n, d in shapes_of(rows):
        by = {r["kernel"]: r for r in rows if int(r["n"]) == n and int(r["d"]) == d}

        def ratio(a: str, b: str) -> str:
            ta, tb = by.get(a, {}).get("median_us"), by.get(b, {}).get("median_us")
            return f"{float(ta) / float(tb):.2f}×" if ta and tb else "—"

        out.append(f"| {n}×{d} | {ratio('v0', 'v1')} | {ratio('v1', 'v2')} | "
                   f"{ratio('v0', 'v2')} | {ratio('torch_softmax', 'v2')} |")
    return "\n".join(out)


def bytes_table(rows: list[dict], kernels: list[str]) -> str:
    head = ("| shape | kernel | DRAM read | DRAM write | L2 total | × ideal | GB/s | "
            "% of peak | source |")
    sep = "|" + "---|" * 9
    if not rows:
        body = [f"| 4096×8192 | {k} |" + " — |" * 7 for k in kernels]
        return "\n".join([head, sep, *body])

    def mb(v: str) -> str:
        return f"{int(v) / 1e6:,.1f} MB" if v else "—"

    out = [head, sep]
    for n, d in shapes_of(rows):
        for k in kernels:
            r = next((x for x in rows
                      if int(x["n"]) == n and int(x["d"]) == d and x["kernel"] == k), None)
            if not r:
                continue
            out.append(
                f"| {n}×{d} | {k} | {mb(r['dram_read_bytes'])} | {mb(r['dram_write_bytes'])} "
                f"| {mb(r['l2_total_bytes'])} | {cell(r['dram_vs_ideal'], '×')} "
                f"| {cell(r['dram_gbps'])} | {cell(r['pct_peak'], '%')} | {r['source']} |")
    if len(out) == 2:
        # bytes.csv exists but holds nothing for these kernels -- the realistic case is
        # `make profile` without `make profile-variants`. Emit placeholders rather than
        # a header with no body, which renders as a broken table.
        out += [f"| 4096×8192 | {k} |" + " — |" * 7 for k in kernels]
    return "\n".join(out)


def env_table() -> str:
    path = RESULTS / "env.json"
    info = json.loads(path.read_text()) if path.exists() else {}

    def g(key: str, suffix: str = "") -> str:
        v = info.get(key)
        return f"{v}{suffix}" if v not in (None, "") else "—"

    gpu = g("gpu")
    if info.get("gpu_memory_gb"):
        gpu += f", {info['gpu_memory_gb']} GB"
    clocks = "—"
    if info.get("clocks_sm_clock"):
        clocks = (f"SM {info['clocks_sm_clock']} (max {g('clocks_sm_clock_max')}), "
                  f"mem {g('clocks_mem_clock')}")
    return "\n".join([
        "| | |", "|---|---|",
        f"| GPU | {gpu} |",
        f"| Peak memory bandwidth | {g('peak_bw_gbs', ' GB/s')} — {g('peak_bw_source')} |",
        f"| Driver / CUDA runtime | {g('driver')} / {g('cuda_runtime')} |",
        f"| PyTorch | {g('torch')} |",
        f"| Clocks | {clocks} |",
        "| Timing | 50 warmup + 200 timed launches, CUDA events, median |",
        f"| Git commit | {g('git_commit')} |",
    ])


def render(summary: list[dict], byte_rows: list[dict]) -> dict[str, str]:
    return {
        "headline-8192": headline_table(summary, 4096, 8192),
        "headline-1024": headline_table(summary, 4096, 1024),
        "timings": timings_table(summary),
        "speedups": speedup_table(summary),
        "bytes": bytes_table(byte_rows, LADDER),
        "bytes-variants": bytes_table(byte_rows, VARIANTS),
        "environment": env_table(),
    }


def splice(text: str, name: str, block: str) -> tuple[str, bool]:
    # Matches an empty region as well as a populated one, so a freshly added pair of
    # markers fills in on the first run rather than being reported as missing.
    pattern = re.compile(
        rf"(<!-- BEGIN:{re.escape(name)} -->\n).*?(<!-- END:{re.escape(name)} -->)",
        re.S)
    if not pattern.search(text):
        print(f"warning: README has no <!-- BEGIN:{name} --> marker; skipping")
        return text, False
    new = pattern.sub(lambda m: m.group(1) + block + "\n" + m.group(2), text)
    return new, new != text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--summary", default=str(RESULTS / "summary.csv"))
    ap.add_argument("--bytes", default=str(RESULTS / "bytes.csv"))
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the README is out of date with results/")
    ap.add_argument("--stdout", action="store_true", help="print tables, write nothing")
    args = ap.parse_args()

    summary = read_csv(Path(args.summary))
    byte_rows = read_csv(Path(args.bytes))
    tables = render(summary, byte_rows)

    if args.stdout:
        for name, block in tables.items():
            print(f"\n## {name}\n\n{block}")
        return 0

    path = Path(args.readme)
    text = original = path.read_text()
    changed_any = False
    for name, block in tables.items():
        text, changed = splice(text, name, block)
        changed_any = changed_any or changed

    if args.check:
        if text != original:
            print("README tables are out of date -- run: python bench/make_tables.py")
            return 1
        print("README tables match results/")
        return 0

    if changed_any:
        path.write_text(text)
        print(f"updated {path}")
    else:
        print(f"{path} already up to date")
    n_timed = len([r for r in summary if r.get("median_us")])
    print(f"  sources: {len(summary)} summary rows ({n_timed} timed), "
          f"{len(byte_rows)} byte rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
