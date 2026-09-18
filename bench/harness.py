"""Timing protocol and correctness check. Written before the kernels, so that each
kernel is a small testable unit dropped into an already-working rig.

Protocol, fixed for every measurement in the project:
    CUDA events, 50 warmup launches discarded, 200 timed launches, report the MEDIAN.

Median, not best-of-N: best-of-N reports the luckiest interaction with the clock
governor and hides variance. p5 and p95 are logged alongside so the spread is visible,
and every individual sample is written to results/raw/ so the distribution can be
checked rather than trusted.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from bench.baselines import reference_fp64

WARMUP = 50
ITERS = 200
TOLERANCE = 1e-5  # max absolute error against the FP64 reference


@dataclass
class Timing:
    median_us: float
    p5_us: float
    p95_us: float
    mean_us: float
    stdev_us: float
    iters: int
    warmup: int
    samples: list[float] = field(repr=False, default_factory=list)


def time_fn(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
            warmup: int = WARMUP, iters: int = ITERS) -> Timing:
    """Median wall time of `fn(x)` in microseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)  # ms -> us

    ordered = sorted(samples)
    return Timing(
        median_us=statistics.median(ordered),
        p5_us=float(np.percentile(ordered, 5)),
        p95_us=float(np.percentile(ordered, 95)),
        mean_us=statistics.fmean(ordered),
        stdev_us=statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        iters=iters,
        warmup=warmup,
        samples=samples,
    )


def max_abs_err(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> float:
    """Max absolute error of `fn(x)` against the FP64 CPU reference.

    NaNs count as failure: they come back as inf so a NaN-producing kernel can never
    pass the tolerance check by accident.
    """
    got = fn(x)
    torch.cuda.synchronize()
    got_np = got.detach().to(torch.float64).cpu().numpy()
    ref = reference_fp64(x)
    if got_np.shape != ref.shape:
        return float("inf")
    diff = np.abs(got_np - ref)
    if not np.isfinite(diff).all():
        return float("inf")
    return float(diff.max())


def validate(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
             tol: float = TOLERANCE) -> tuple[bool, float]:
    """(passed, max_abs_err). A kernel that fails does not get timed.

    Exceptions propagate on purpose: a launch failure is a different kind of event from
    a wrong answer, and callers distinguish the two (v1 has a documented shared-memory
    ceiling that should read as a limit, not as a bad result).
    """
    err = max_abs_err(fn, x)
    return err <= tol, err


def write_raw(path: Path, timing: Timing, header: list[str]) -> None:
    """Every individual sample, so the distribution is checkable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in header:
            f.write(line + "\n")
        f.write("iteration,microseconds\n")
        for i, us in enumerate(timing.samples):
            f.write(f"{i},{us:.4f}\n")


def make_input(n: int, d: int, seed: int = 0, device: str = "cuda") -> torch.Tensor:
    """Fixed-seed input. Every kernel and the reference see exactly the same bytes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float32).to(device).contiguous()
