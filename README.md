# fusion-bench

Three softmax kernels forming a ladder — multi-pass, fused, single-pass — where each rung
removes one source of memory traffic, and Nsight Compute counters show whether the
measured speedup equals the measured reduction in DRAM bytes.

The claim, in one sentence: **v1 moves N× fewer bytes than v0 and is N× faster, so the
speedup is the traffic reduction.** Everything in this repo exists to support or refute
that sentence with data.

> **Status: kernels and harness complete, numbers pending.** The extension compiles,
> links and imports without a GPU — `scripts/build_check.sh` runs the real
> `cpp_extension.load()` path in a container, and `scripts/nvcc_check.sh` reports
> `sm_80` register usage (zero spills). Every number below still needs a GPU. Every table below is
> generated from `results/summary.csv` and `results/bytes.csv`; the em dashes are filled
> in by a single GPU run (see [Reproducing](#reproducing)). Predictions were written
> down *before* profiling in [`docs/notes.md`](docs/notes.md) so they can be wrong in
> public.

---

## The ladder

| | what changed | global passes over the row | modelled bytes |
|---|---|---|---|
| **v0** naive | max, sum, normalize as three separate passes, all from global memory | 3 read + 1 write | `16·N·D` |
| **v1** fused | the row is staged in shared memory once; both reductions run there | 1 read + 1 write | `8·N·D` |
| **v2** online | one pass with a running max and running sum; warp-shuffle reductions; `float4` loads; the row stays in registers | 1 read + 1 write | `8·N·D` |

v0 is deliberately bad but not *stupidly* bad: threads within a block read consecutive
addresses, so its loads are already coalesced. That matters — it makes the v0 → v1 delta
a measurement of fusion alone, not fusion confounded with access pattern.

Phase 5's three changes are also kept as separate kernels — `v2a` (online pass only),
`v2b` (`+` warp shuffles), `v2c` (`+` `float4` and register residency) — so the rung-2
win can be attributed to the change that produced it rather than to all three at once.
`make bench-variants` times them, `make profile-variants` gets their counters, and a
second chart is drawn automatically. The prediction, recorded
before measuring, is that **v2a loses to v1** and almost all of v2's win is v2c's
register residency; see [`docs/notes.md`](docs/notes.md).

v2's online update is Milakov & Gimelshein ([arXiv:1805.02867](https://arxiv.org/abs/1805.02867)):

```
m_new = max(m_old, x)
d_new = d_old · exp(m_old − m_new) + exp(x − m_new)
```

the same rescaling trick that makes FlashAttention work.

## The charts

The claim itself: for each rung, the reduction in bytes moved beside the measured
speedup. Equal pairs mean the speedup *is* the traffic reduction. A short orange bar
beside a tall blue one means bytes were saved that did not buy any time — which is what
4096×1024 is expected to show, because its working set fits in L2 and the reads v0
"saved" were already cache hits.

![traffic reduction vs measured speedup](results/plots/claim.png#gh-light-mode-only)
![traffic reduction vs measured speedup](results/plots/claim_dark.png#gh-dark-mode-only)

Median time per version, with the two torch references drawn as lines:

![median time per version](results/plots/time.png#gh-light-mode-only)
![median time per version](results/plots/time_dark.png#gh-dark-mode-only)

And where v2's win actually comes from, one change per rung:

![v2 attribution](results/plots/v2_attribution.png#gh-light-mode-only)
![v2 attribution](results/plots/v2_attribution_dark.png#gh-dark-mode-only)

## Timings

50 warmup launches discarded, 200 timed launches, CUDA events, **median** (not best-of-N).
p5/p95 in the table, every individual sample in `results/raw/`.

<!-- BEGIN:timings -->
| shape | v0 | v1 | v2 | `torch_softmax` | `torch_compile` |
|---|---|---|---|---|---|
| 4096×1024 | — | — | — | — | — |
| 4096×8192 | — | — | — | — | — |
<!-- END:timings -->

Every ratio the claim above rests on, computed from those medians rather than asserted:

<!-- BEGIN:speedups -->
| shape | v0 → v1 | v1 → v2 | v0 → v2 | v2 vs `torch.softmax` |
|---|---|---|---|---|
| 4096×1024 | — | — | — | — |
| 4096×8192 | — | — | — | — |
<!-- END:speedups -->

## The byte table

Measured with `ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum`
against a single-shot script — one launch per kernel, never the benchmark loop.

<!-- BEGIN:bytes -->
| shape | kernel | DRAM read | DRAM write | L2 total | × ideal | GB/s | % of peak | source |
|---|---|---|---|---|---|---|---|---|
| 4096×8192 | v0 | — | — | — | — | — | — | — |
| 4096×8192 | v1 | — | — | — | — | — | — | — |
| 4096×8192 | v2 | — | — | — | — | — | — | — |
<!-- END:bytes -->

And the same counters for the attribution ladder, which is where the rung-2 claim is
settled — v2a and v2b should show 1.5× ideal, v2c 1.0×:

<!-- BEGIN:bytes-variants -->
| shape | kernel | DRAM read | DRAM write | L2 total | × ideal | GB/s | % of peak | source |
|---|---|---|---|---|---|---|---|---|
| 4096×8192 | v2a | — | — | — | — | — | — | — |
| 4096×8192 | v2b | — | — | — | — | — | — | — |
| 4096×8192 | v2c | — | — | — | — | — | — | — |
<!-- END:bytes-variants -->

"Ideal" is one read plus one write of the tensor, the floor for any softmax. Achieved
bandwidth is *measured* bytes divided by median time, expressed against the card's stated
peak — state the SKU, because an A100 80GB PCIe (1935 GB/s) and an SXM4 (2039 GB/s) are
not the same denominator.

## What the two shapes are for

**4096×1024** is 16.8 MB in FP32 and fits inside the A100's 40 MB L2. v0's extra passes
should mostly hit L2 rather than DRAM, so the DRAM counters will show far less traffic
than the algorithm implies and the "N× fewer bytes, N× faster" relationship is expected
to *break* here. That is why `lts__t_bytes.sum` is captured alongside the DRAM metrics:
the claim is that the traffic reduction is real but happens one level up the hierarchy,
and an L2 counter demonstrates that where an argument would only assert it.

**4096×8192** is 134 MB, well past L2, and is where the model should hold.

## Verdict

— *(one sentence tying the speedup to the traffic reduction, or explaining why it does
not hold at the small shape)*

## Limitations

Forward only, FP32 only, 2D tensors, last dimension, no autograd, no backward pass, no
mixed precision, no strided or non-contiguous input. `torch.softmax` is general across all
of those. **This is a like-for-like comparison on the forward pass at two shapes and is
not a claim of beating PyTorch.**

Two further caveats, stated rather than buried:

- v2 has a generic fallback path (scalar, re-reads the row) for `D` that is not a
  multiple of 4 or is too long to hold in registers. Both benchmark shapes take the fast
  register-resident path; every CSV row records which path ran.
- A fully masked row (every element `−inf`) is undefined input. These kernels return a
  uniform `1/D`; `torch.softmax` returns `NaN`. The case is documented in
  [`docs/notes.md`](docs/notes.md) and excluded from the cross-check rather than silently
  differing.

## Correctness

Reference is NumPy **float64**, computed from the same fixed-seed input. Pass condition is
max absolute error below `1e-5`. **A kernel that fails does not get timed.**

Accuracy is a gate, not an achievement: every version computes the same answer, so the
chart below exists to answer the obvious objection to a speedup — that precision was
traded for it. Distance from the tolerance line is headroom.

![accuracy headroom](results/plots/accuracy.png#gh-light-mode-only)
![accuracy headroom](results/plots/accuracy_dark.png#gh-dark-mode-only)

`tests/verify.py` covers the cases that actually break softmax kernels:

| case | what it exercises |
|---|---|
| `D = 1` | degenerate row; almost every thread in the block is idle |
| `D % 32 ≠ 0` | warp remainder in the reduction |
| `D % 4 ≠ 0` | `float4` tail handling |
| a row containing `−inf` | masked positions, as in causal attention |
| `x + 1000` | large magnitude — overflows without the max subtraction |

The last one is included deliberately: it is the case a naive implementation silently
fails, and the test prints the overflow it avoids (`sum(exp(x))` → `inf` versus
`sum(exp(x − max x))` → a finite number) so the reason for the subtraction is visible
rather than assumed.

## Reproducing

```bash
# 0a. no GPU needed: simulate the kernels' reduction logic against the FP64 reference
make sim

# 0b. on the rented box, counter access FIRST — the assumption the project rests on
bash scripts/phase0_check.sh

# 1. compile the extension and round-trip a copy kernel
make build

# 2. correctness before speed
make verify

# 3. timings -> results/summary.csv, results/raw/
#    (bench-variants also times v2a/v2b/v2c, which step 4 profiles)
make bench-variants

# 4. counters -> results/raw/ncu_*.csv -> results/bytes.csv
make profile && make profile-variants && make bytes

# 5. chart -> results/plots/
make plot

# 6. fill in every table in this README from the CSVs -- no number typed by hand
make tables
```

Total GPU time for the measurement run is under ten minutes. Develop on a cheap card and
move to the A100 only for the numbers that go in this file. There is a step-by-step
session guide in [`docs/runbook.md`](docs/runbook.md).

**If Nsight counters are blocked** (many cloud containers and every Colab runtime refuse
them), `make analytic` derives the byte table from the algorithm instead. That is weaker
evidence and the table is labelled `derived` in its `source` column — say so plainly here
too rather than leaving it ambiguous.

## Environment

Every generated CSV carries this block as comment lines above its header, because a
number without its environment is not reproducible.

<!-- BEGIN:environment -->
| | |
|---|---|
| GPU | — |
| Peak memory bandwidth | — — — |
| Driver / CUDA runtime | — / — |
| PyTorch | — |
| Clocks | — |
| Timing | 50 warmup + 200 timed launches, CUDA events, median |
| Git commit | — |
<!-- END:environment -->

## Layout

```
src/
  kernels.cu         v0, v1, v2 + the v2a/v2b/v2c attribution variants
  bindings.cpp       torch extension entry points, all validation
  softmax.h          launcher declarations (kernels.cu stays torch-free)
bench/
  ext.py             JIT build via torch.utils.cpp_extension.load
  make_tables.py     regenerates the tables in this file from results/
  kernels.py         the kernel registry — ladder and attribution variants
  harness.py         timing protocol — events, warmup, median, raw logging
  env.py             environment capture, written into every CSV header
  baselines.py       naive composition, torch.softmax, torch.compile, FP64 reference
  bytes_model.py     the analytic traffic model (prediction, and Plan B)
  run_all.py         validate + time -> results/summary.csv
  profile_one.py     single-shot launches, for ncu only
  parse_ncu.py       ncu CSV -> results/bytes.csv
  plot.py            charts -> results/plots/ (claim, time, attribution, accuracy)
tests/
  verify.py          FP64 cross-check on the real kernels, all edge cases
  sim_logic.py       CPU simulation of the reduction logic — no GPU required
scripts/
  phase0_check.sh    GPU + counter access verification
  nvcc_check.sh      compile kernels.cu with real nvcc in Docker — no GPU needed
  build_check.sh     compile + link + import the extension in Docker — no GPU needed
  profile.sh         the ncu invocation
  lock_clocks.sh     clock locking, if permitted
docs/
  notes.md           prediction, measurement, verdict per rung
  runbook.md         the GPU session, start to finish
```

## Relationship to CUDA-SGEMM-Optimization

[That repo](https://github.com/Sanjith-Shan/CUDA-SGEMM-Optimization) already contains a
fused, `float4`-vectorized softmax kernel benchmarked against torch. This is not a
rewrite of it. The delta:

| already exists there | what this adds |
|---|---|
| one final fused kernel | a three-rung ladder, so the win is attributable |
| two-pass max/sum | online softmax — single pass, running max |
| GB/s derived from timing | measured `dram__bytes_*` from Nsight counters |
| counters unavailable on that box | counter access verified before any code was written |
| "~70–75% of HBM peak" asserted | achieved bandwidth computed from measured traffic |

If the counters work here, backporting the byte table into that repo's fused section is
worth doing afterwards.

## References

- Milakov & Gimelshein, *Online normalizer calculation for softmax*, [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- Dao et al., *FlashAttention*, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- NVIDIA Nsight Compute CLI documentation — metric names change between versions;
  `scripts/phase0_check.sh` verifies them against `ncu --query-metrics` on the actual
  machine rather than trusting a copied command line.
