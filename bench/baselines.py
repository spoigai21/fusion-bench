"""The things the kernels are measured against.

1. naive_composition  -- genuinely multi-kernel: max, sub, exp, sum, div are separate
                         ATen ops, each materializing an intermediate. This is the bar
                         v0 should already clear.
2. torch.softmax      -- ATen's fused kernel. THE REAL BAR. For these shapes it uses a
                         persistent warp-per-row kernel and is hard to beat.
3. torch.compile      -- inductor fusing the naive composition automatically.
4. numpy float64      -- the correctness reference. Not a speed baseline; never timed.
"""

from __future__ import annotations

import numpy as np
import torch


def naive_composition(x: torch.Tensor) -> torch.Tensor:
    """Softmax written the obvious way. Several kernel launches and round trips."""
    e = (x - x.max(-1, keepdim=True).values).exp()
    return e / e.sum(-1, keepdim=True)


def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


_compiled = None


def torch_compile_softmax(x: torch.Tensor) -> torch.Tensor:
    """naive_composition run through inductor. Compiled lazily and cached, so the
    first call pays compilation -- always inside warmup, never inside the timed loop."""
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(naive_composition, dynamic=False)
    return _compiled(x)


def reset_compile_cache() -> None:
    """Force recompilation. Needed between shapes: with dynamic=False, inductor
    specializes on the shape it first saw, and a silent recompile inside a timed loop
    would land entirely in one sample."""
    global _compiled
    _compiled = None


def reference_fp64(x: torch.Tensor) -> np.ndarray:
    """The ground truth: float64 softmax on CPU, computed from the same input tensor.

    Rows that are entirely -inf are left as-is (they produce NaN here and uniform 1/D in
    the kernels); the verifier does not generate them. See docs/notes.md.
    """
    a = x.detach().to(torch.float64).cpu().numpy()
    m = a.max(axis=-1, keepdims=True)
    e = np.exp(a - m)
    return e / e.sum(axis=-1, keepdims=True)


# name -> callable. Order is the order rows appear in the CSV.
BASELINES = {
    "naive_composition": naive_composition,
    "torch_softmax": torch_softmax,
    "torch_compile": torch_compile_softmax,
}

# Drawn as horizontal reference lines on the chart rather than as bars.
REFERENCE_LINES = ("torch_softmax", "torch_compile")
