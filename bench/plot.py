#!/usr/bin/env python3
"""The chart: median time per version, with torch.softmax and torch.compile as
reference lines. Reads results/summary.csv, writes results/plots/.

Light and dark versions are both rendered, because the README is read in both themes
on GitHub and an automatic flip of a light chart is not a dark chart.

    python bench/plot.py
    python bench/plot.py --summary results/summary.csv --outdir results/plots
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

LADDER = ["v0", "v1", "v2"]
LADDER_LABELS = {
    "v0": "v0  naive\n3 global passes",
    "v1": "v1  fused\nshared memory",
    "v2": "v2  online\nwarp shuffle + float4",
    # the attribution ladder inside v2, drawn as a second chart when those rows exist
    "v2a": "v2a  online only\ntree reduction",
    "v2b": "v2b  + warp\nshuffle",
    "v2c": "v2c  + float4\n+ registers",
}
# v1 is the starting point of the breakdown: the question it answers is what each of
# v2's three changes bought over the rung below it.
ATTRIBUTION = ["v1", "v2a", "v2b", "v2c"]

# The pairs the central claim is made about: for each, is the speedup equal to the
# reduction in bytes moved?
CLAIM_PAIRS = [("v0", "v1"), ("v1", "v2"), ("v0", "v2")]
# Drawn as reference lines rather than bars: they are the bar to clear, not rungs.
# Distinct dash patterns as well as distinct hues, so identity never rests on color.
REFS = [
    ("torch_softmax", "torch.softmax (ATen fused)", (0, (6, 3))),
    ("torch_compile", "torch.compile", (0, (2, 2))),
]

# Validated categorical slots 1-3 (see the dataviz palette reference); both modes are
# selected, not flipped. Slot 1 carries the kernels, 2 and 3 the two torch references.
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=("#2a78d6", "#eb6834", "#1baf7a")),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series=("#3987e5", "#d95926", "#199e70")),
}


def load_bytes(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    """dram_total_bytes keyed by shape then kernel. Empty if the file is absent."""
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    out: dict[tuple[int, int], dict[str, float]] = {}
    for r in csv.DictReader(lines):
        if r.get("dram_total_bytes"):
            out.setdefault((int(r["n"]), int(r["d"])), {})[r["kernel"]] = float(
                r["dram_total_bytes"])
    return out


def draw_claim(times, byte_data, mode: str, out: Path) -> Path:
    """The thesis chart: traffic reduction beside measured speedup, for each rung.

    Equal-height pairs mean the speedup is the traffic reduction and the model holds.
    A short speedup bar next to a tall traffic bar means bytes were saved that did not
    buy time -- which is what the 4096x1024 shape is expected to show, because its
    working set fits in L2 and the 'saved' reads were already cache hits.
    """
    c = THEME[mode]
    shapes = sorted(set(times) & set(byte_data))
    fig, axes = plt.subplots(
        1, len(shapes), figsize=(5.6 * len(shapes), 4.2), facecolor=c["surface"])
    if len(shapes) == 1:
        axes = [axes]

    width = 0.34   # leaves a small gap between the paired bars
    for ax, shape in zip(axes, shapes):
        n, d = shape
        t, b = times[shape], byte_data[shape]
        labels, traffic, speedup = [], [], []
        for a, bb in CLAIM_PAIRS:
            if a in t and bb in t and a in b and bb in b and t[bb] and b[bb]:
                labels.append(f"{a} → {bb}")
                traffic.append(b[a] / b[bb])
                speedup.append(t[a] / t[bb])

        ax.set_facecolor(c["surface"])
        x = list(range(len(labels)))
        ax.bar([i - width / 2 for i in x], traffic, width, color=c["series"][0],
               label="bytes moved: fewer by", zorder=3)
        ax.bar([i + width / 2 for i in x], speedup, width, color=c["series"][1],
               label="time: faster by", zorder=3)

        top = max(traffic + speedup + [1.0])
        for i, (tr, sp) in enumerate(zip(traffic, speedup)):
            for off, v in ((-width / 2, tr), (width / 2, sp)):
                ax.text(i + off, v + top * 0.02, f"{v:.2f}×", ha="center", va="bottom",
                        color=c["ink"], fontsize=9.5, zorder=4)

        # 1.0x is "no change at all" -- the line a bar has to clear to mean anything.
        ax.axhline(1.0, color=c["axis"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, color=c["ink2"])
        ax.set_ylim(0, top * 1.22)
        ax.tick_params(axis="y", colors=c["muted"], labelsize=9)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("ratio (higher is better)", color=c["muted"], fontsize=9)
        ax.set_title(f"{n}×{d}   ({n * d * 4 / 1e6:.1f} MB input)",
                     color=c["ink"], fontsize=11, pad=10, loc="left")
        ax.grid(axis="y", color=c["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])

    handles, lbls = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, lbls, loc="lower center", ncol=2, frameon=False,
                     fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    for text in leg.get_texts():
        text.set_color(c["ink2"])

    fig.suptitle("Is the speedup the traffic reduction?  —  equal pairs mean yes",
                 color=c["ink"], fontsize=13, x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=c["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def load_summary(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    data: dict[tuple[int, int], dict[str, float]] = {}
    for r in csv.DictReader(lines):
        if not r.get("median_us"):
            continue
        data.setdefault((int(r["n"]), int(r["d"])), {})[r["kernel"]] = float(r["median_us"])
    return data


def draw(data, mode: str, out: Path, bars: list[str] | None = None,
         title: str = "Softmax kernel ladder — median time per version") -> Path:
    c = THEME[mode]
    bars = bars or LADDER
    shapes = sorted(data.keys())
    fig, axes = plt.subplots(
        1, len(shapes), figsize=(5.6 * len(shapes), 4.0), facecolor=c["surface"]
    )
    if len(shapes) == 1:
        axes = [axes]

    for ax, shape in zip(axes, shapes):
        n, d = shape
        times = data[shape]
        vals = [times.get(k, 0.0) for k in bars]
        ypos = list(range(len(bars)))[::-1]  # first rung at the top, reads downward

        ax.set_facecolor(c["surface"])
        ax.barh(ypos, vals, height=0.42, color=c["series"][0], zorder=3)

        # Direct labels: 3 marks, so every bar gets its value. Text wears ink tokens,
        # never the series color.
        span = max(vals + [t for k, _, _ in REFS if (t := times.get(k))]) or 1.0
        base_key = bars[0]
        for y, k, v in zip(ypos, bars, vals):
            if not v:
                continue
            label = f"{v:,.0f} µs"
            if times.get(base_key) and k != base_key:
                label += f"   {times[base_key] / v:.2f}× {base_key}"
            # A halo in the surface colour keeps the label legible where it crosses
            # a reference line.
            ax.text(v + span * 0.02, y, label, va="center", ha="left",
                    color=c["ink"], fontsize=9.5, zorder=5,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground=c["surface"])])

        for i, (key, label, dashes) in enumerate(REFS):
            t = times.get(key)
            if not t:
                continue
            ax.axvline(t, color=c["series"][i + 1], linewidth=2, linestyle=dashes,
                       zorder=2, label=label)

        ax.set_yticks(ypos)
        ax.set_yticklabels([LADDER_LABELS[k] for k in bars], fontsize=9, color=c["ink2"])
        ax.tick_params(axis="x", colors=c["muted"], labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, span * 1.32)
        ax.set_xlabel("median time (µs) — lower is better", color=c["muted"], fontsize=9)
        ax.set_title(f"{n}×{d}   ({n * d * 4 / 1e6:.1f} MB input)",
                     color=c["ink"], fontsize=11, pad=10, loc="left")

        ax.grid(axis="x", color=c["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])
            ax.spines[side].set_linewidth(1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        leg = fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                         frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
        for text in leg.get_texts():
            text.set_color(c["ink2"])

    fig.suptitle(title, color=c["ink"], fontsize=13, x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=c["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "summary.csv"))
    ap.add_argument("--bytes", default=str(ROOT / "results" / "bytes.csv"))
    ap.add_argument("--outdir", default=str(ROOT / "results" / "plots"))
    args = ap.parse_args()

    path = Path(args.summary)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run bench/run_all.py first")

    data = load_summary(path)
    if not data:
        raise SystemExit(f"{path} has no timed rows")

    byte_data = load_bytes(Path(args.bytes))
    outdir = Path(args.outdir)
    for mode in ("light", "dark"):
        suffix = "" if mode == "light" else "_dark"
        print(f"wrote {draw(data, mode, outdir / f'time{suffix}.png')}")

    # The breakdown chart is drawn only when the variant rows are actually present,
    # i.e. after a --variants run. It answers a different question from the headline
    # chart, so it gets its own figure rather than three more bars on that one.
    if any(k in t for t in data.values() for k in ("v2a", "v2b", "v2c")):
        for mode in ("light", "dark"):
            suffix = "" if mode == "light" else "_dark"
            out = draw(data, mode, outdir / f"v2_attribution{suffix}.png",
                       bars=ATTRIBUTION,
                       title="Where v2's win comes from — one change per rung")
            print(f"wrote {out}")

    # The claim chart needs both timings and counters, so it appears only once the
    # profiler has run. Without it the project's central assertion has no picture.
    if byte_data:
        for mode in ("light", "dark"):
            suffix = "" if mode == "light" else "_dark"
            print(f"wrote {draw_claim(data, byte_data, mode, outdir / f'claim{suffix}.png')}")
    else:
        print("(no results/bytes.csv yet -- skipping the claim chart; run make profile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
