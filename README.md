# Fusion Bench

Three softmax kernels forming a ladder — multi-pass, fused, single-pass — where each rung
removes one source of memory traffic, and Nsight Compute counters show whether the
measured speedup equals the measured reduction in DRAM bytes.

The claim, in one sentence: **v1 moves N× fewer bytes than v0 and is N× faster, so the
speedup is the traffic reduction.** Everything in this repo exists to support or refute
that sentence with data.

Run on an **A100-SXM4-40GB**. Every number below is generated from `results/` by
`make tables`; none is typed by hand.

## Headline result

Forward softmax over the last dimension of a 4096×8192 FP32 tensor:

<!-- BEGIN:headline-8192 -->
| Rung | median | effective GB/s | % of HBM peak | Speedup vs prev | vs `torch.softmax` |
|---|---|---|---|---|---|
| `softmax_v0` naive | 373.8 µs | 718.2 | 46.2% | — | 0.65× |
| `softmax_v1` fused | 223.2 µs | 1202.5 | 77.3% | 1.67× | 1.09× |
| `softmax_v2` online | **209.9 µs** | 1278.8 | 82.2% | **1.06×** | **1.16×** |
| `torch.softmax` (ATen) | 242.7 µs | 1106.1 | 71.1% | — | 1.00× |
| `torch.compile` | 262.1 µs | 1024.0 | 65.9% | — | 0.93× |
| naive composition (multi-kernel) | 885.8 µs | 303.1 | 19.5% | — | 0.27× |
<!-- END:headline-8192 -->

![traffic reduction vs measured speedup](results/plots/claim.png#gh-light-mode-only)
![traffic reduction vs measured speedup](results/plots/claim_dark.png#gh-dark-mode-only)

**v0 → v2 moved 1.82× fewer bytes and ran 1.78× faster — the speedup *is* the traffic
reduction, within 2%.** That is the whole claim, measured with Nsight hardware counters
rather than inferred from timing. The chart above is the test: blue is bytes saved,
orange is time saved, and equal pairs mean the model holds.

The left panel is the more interesting half. At 4096×1024 the model **inverts** — v1
moves fewer bytes and runs *slower* — because the 16.8 MB working set fits in the A100's
40 MB L2. [Where the model breaks](#where-the-model-breaks) has the counters.

---

## The ladder — one idea per rung

| | what changed | global passes | modelled bytes |
|---|---|---|---|
| **v0** naive | max, sum and normalize as three separate passes, all from global memory | 3 read + 1 write | `16·N·D` |
| **v1** fused | the row is staged in shared memory once; both reductions run there | 1 read + 1 write | `8·N·D` |
| **v2** online | one pass with a running max and running sum; warp-shuffle reductions; `float4` loads; the row stays in registers | 1 read + 1 write | `8·N·D` |

v0 is deliberately bad but not *stupidly* bad: threads within a block read consecutive
addresses, so its loads are already coalesced. That makes the v0 → v1 delta a
measurement of fusion alone, not fusion confounded with access pattern.

v2's online update is Milakov & Gimelshein ([arXiv:1805.02867](https://arxiv.org/abs/1805.02867)):

```
m_new = max(m_old, x)
d_new = d_old · exp(m_old − m_new) + exp(x − m_new)
```

the same rescaling trick that makes FlashAttention work.

![median time per version](results/plots/time.png#gh-light-mode-only)
![median time per version](results/plots/time_dark.png#gh-dark-mode-only)

At the smaller shape the ordering changes completely:

<!-- BEGIN:headline-1024 -->
| Rung | median | effective GB/s | % of HBM peak | Speedup vs prev | vs `torch.softmax` |
|---|---|---|---|---|---|
| `softmax_v0` naive | 45.1 µs | 744.7 | 47.9% | — | 0.80× |
| `softmax_v1` fused | 48.1 µs | 697.2 | 44.8% | 0.94× | 0.74× |
| `softmax_v2` online | **33.8 µs** | 993.0 | 63.9% | **1.42×** | **1.06×** |
| `torch.softmax` (ATen) | 35.8 µs | 936.2 | 60.2% | — | 1.00× |
| `torch.compile` | 92.2 µs | 364.1 | 23.4% | — | 0.39× |
| naive composition (multi-kernel) | 126.0 µs | 266.4 | 17.1% | — | 0.28× |
<!-- END:headline-1024 -->

## Where v2's win comes from

v2 bundles three changes, so they are also kept as separate kernels — `v2a` (online pass
only), `v2b` (`+` warp shuffles), `v2c` (`+` `float4` and register residency) — and timed
and profiled independently. The prediction, [recorded before
measuring](docs/notes.md), was that **v2a would lose to v1** and that almost all of the
win would be v2c's register residency.

![v2 attribution](results/plots/v2_attribution.png#gh-light-mode-only)
![v2 attribution](results/plots/v2_attribution_dark.png#gh-dark-mode-only)

| at 4096×8192 | median | DRAM vs v1 | vs v1 |
|---|---|---|---|
| v1 fused | 223.2 µs | 1.00× | 1.00× |
| v2a online only | 300.0 µs | **1.51×** | 0.74× |
| v2b `+` warp shuffle | 300.0 µs | 1.49× | 0.74× |
| v2c `+` float4 + registers | **212.0 µs** | 1.00× | **1.05×** |

The prediction held, including the unflattering part. **The online formulation on its own
is a regression** — it gives up v1's staged row and has to re-read it, moving 1.51× the
traffic against a predicted 1.5×. **Warp shuffles bought exactly nothing**: v2a and v2b
are both 300.03 µs, identical to the microsecond. The entire rung-2 win is v2c's register
residency, 300.0 → 212.0 µs.

So the honest headline is *the win is register residency, not the online formulation* —
with the online formulation being what **permits** it, since you cannot hold the row in
registers and also make two passes over it.

## Timings

50 warmup launches discarded, 200 timed launches, CUDA events, **median** (not best-of-N).
p5/p95 in the table, every individual sample in `results/raw/`.

<!-- BEGIN:timings -->
| shape | v0 | v1 | v2 | `torch_softmax` | `torch_compile` |
|---|---|---|---|---|---|
| 4096×1024 | 45 µs | 48 µs | 34 µs | 36 µs | 92 µs |
| 4096×8192 | 374 µs | 223 µs | 210 µs | 243 µs | 262 µs |
<!-- END:timings -->

Every ratio the claim above rests on, computed from those medians rather than asserted:

<!-- BEGIN:speedups -->
| shape | v0 → v1 | v1 → v2 | v0 → v2 | v2 vs `torch.softmax` |
|---|---|---|---|---|
| 4096×1024 | 0.94× | 1.42× | 1.33× | 1.06× |
| 4096×8192 | 1.67× | 1.06× | 1.78× | 1.16× |
<!-- END:speedups -->

## The byte table

Measured with `ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum`
against a single-shot script — one launch per kernel, never the benchmark loop.

<!-- BEGIN:bytes -->
| shape | kernel | DRAM read | DRAM write | L2 total | × ideal | GB/s | % of peak | source |
|---|---|---|---|---|---|---|---|---|
| 4096×1024 | v0 | 16.8 MB | 3.8 MB | 54.8 MB | 0.61× | 457.4 | 29.4% | measured |
| 4096×1024 | v1 | 16.8 MB | 1.7 MB | 55.3 MB | 0.55× | 385.0 | 24.8% | measured |
| 4096×1024 | v2 | 16.8 MB | 1.6 MB | 55.3 MB | 0.55× | 545.3 | 35.1% | measured |
| 4096×8192 | v0 | 333.9 MB | 121.0 MB | 995.8 MB | 1.69× | 1217.2 | 78.3% | measured |
| 4096×8192 | v1 | 134.2 MB | 116.2 MB | 525.5 MB | 0.93× | 1122.0 | 72.2% | measured |
| 4096×8192 | v2 | 134.2 MB | 115.8 MB | 525.5 MB | 0.93× | 1191.0 | 76.6% | measured |
<!-- END:bytes -->

And the same counters for the attribution ladder, which is where the rung-2 claim is
settled — v2a and v2b should show 1.5× ideal, v2c 1.0×:

<!-- BEGIN:bytes-variants -->
| shape | kernel | DRAM read | DRAM write | L2 total | × ideal | GB/s | % of peak | source |
|---|---|---|---|---|---|---|---|---|
| 4096×1024 | v2a | 16.8 MB | 3.9 MB | 57.8 MB | 0.62× | 531.8 | 34.2% | measured |
| 4096×1024 | v2b | 16.8 MB | 1.8 MB | 55.9 MB | 0.55× | 502.8 | 32.3% | measured |
| 4096×1024 | v2c | 16.8 MB | 1.6 MB | 55.3 MB | 0.55× | 545.0 | 35.0% | measured |
| 4096×8192 | v2a | 256.3 MB | 121.2 MB | 783.8 MB | 1.41× | 1257.9 | 80.9% | measured |
| 4096×8192 | v2b | 254.7 MB | 119.0 MB | 782.4 MB | 1.39× | 1245.3 | 80.1% | measured |
| 4096×8192 | v2c | 134.2 MB | 115.8 MB | 525.3 MB | 0.93× | 1179.7 | 75.9% | measured |
<!-- END:bytes-variants -->

"Ideal" is one read plus one write of the tensor, the floor for any softmax. Achieved
bandwidth is *measured* bytes divided by median time, expressed against the card's stated
peak — this ran on an A100-SXM4-40GB at 1555 GB/s, which is the denominator for every
"% of peak" figure here.

Two things the counters settle that timing alone could not. At 4096×8192, v0 read
**333.9 MB where its three passes imply 402.7 MB** — L2 already absorbed ~17% of the
re-reads even at a size well past the cache. And v2a/v2b measured **1.41×/1.39× ideal**
against a predicted 1.50×, for the same reason.

## Where the model breaks

**4096×1024 is 16.8 MB in FP32 and fits inside the A100's 40 MB L2**, and the result is
not a weaker correlation — it is an inverted one. v1 moves 1.11× fewer bytes than v0 and
runs **0.94×, slower**.

The counters show why, which is exactly what `lts__t_bytes.sum` was captured for:

| at 4096×1024 | DRAM total | L2 total | × ideal |
|---|---|---|---|
| v0 | 20.6 MB | 54.8 MB | 0.61× |
| v1 | 18.5 MB | 55.3 MB | 0.55× |

v0 moved 20.6 MB of DRAM traffic, not the 67 MB its three passes imply. Both kernels sit
**below even the one-read-one-write floor** — 0.61× and 0.55× of "ideal" — because L2
absorbs the writes too, and the L2 column carries roughly 2.7× the DRAM traffic. The
reads v1 "saved" were already cache hits, so it paid for shared-memory staging and got
nothing back for it.

**A traffic model that ignores the cache hierarchy predicts the wrong sign here, not
merely the wrong magnitude.** That is the most useful thing in this repository, and it is
why the small shape was kept rather than quietly dropped.

**4096×8192** is 134 MB, well past L2, and is where the model holds.

## Verdict

**At 4096×8192, v0 → v2 moved 1.82× fewer bytes and ran 1.78× faster — the speedup is
the traffic reduction, within 2%.** v0 → v1 is the same story at 1.82× versus 1.67×.
That is the claim this project set out to test, measured with hardware counters rather
than inferred from timing, and it holds.

**At 4096×1024 the model does not merely weaken — it inverts.** v1 moves 1.11× fewer
bytes than v0 and runs *slower*, 0.94×. The counters say why: v0 moved 20.6 MB of DRAM
traffic, not the 67 MB its three passes imply, because the 16.8 MB input fits inside the
A100's 40 MB L2 (54.8 MB of L2 traffic against 20.6 MB of DRAM). Both kernels are below
even the 1-read-1-write floor — 0.61× and 0.55× of "ideal" — because the cache absorbs
the writes too. The reads v1 "saved" were already cache hits, so it paid for
shared-memory staging and got nothing back. **A traffic model that ignores the cache
hierarchy predicts the wrong sign here, not just the wrong magnitude.**

The sharpest case is v1 → v2 at that shape: **identical DRAM traffic (1.01×) and 1.42×
faster.** Time and bytes decouple completely. Whatever v2 wins there, it is not
bandwidth — it is register residency and `float4` loads avoiding L2 round trips.

On the attribution, the prediction recorded in [`docs/notes.md`](docs/notes.md) before
profiling was right, including the part that was unflattering to the design: **v2a, the
online formulation on its own, loses to v1** (300.0 µs vs 223.2 µs) because it re-reads
the row to normalise and moves 1.51× the traffic. Adding warp shuffles bought **exactly
nothing** — v2a and v2b are both 300.03 µs. The entire rung-2 win is v2c's register
residency, 300.0 → 212.0 µs. The honest headline is *the win is register residency, not
the online formulation* — with the online formulation being what permits it, since you
cannot hold the row in registers and also make two passes over it.

One prediction was wrong and is worth stating plainly. An earlier version of `notes.md`
claimed v1 → v2 would be an occupancy win; reading real `ptxas` output beforehand showed
both sit at 50% occupancy at `D = 8192`, and the correction is in the file. The measured
1.06× at that shape is consistent with the corrected prediction, not the original one.

Against the real bar: **v2 beats `torch.softmax` by 1.16× at 4096×8192 and 1.06× at
4096×1024**, and beats the naive multi-kernel composition by 4.2×. That is a narrower
win than the headline numbers suggest, and it is forward-only FP32 on two shapes — see
Limitations.

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

## Hardware these numbers came from

Every generated CSV carries this block as comment lines above its header, because a
number without its environment is not reproducible.

<!-- BEGIN:environment -->
| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 42.4 GB |
| Peak memory bandwidth | 1555.0 GB/s — table lookup: A100-SXM4-40GB |
| Driver / CUDA runtime | 580.105.08 / 12.8 |
| PyTorch | 2.7.0 |
| Clocks | SM 240 MHz (max 1410 MHz), mem 1215 MHz |
| Timing | 50 warmup + 200 timed launches, CUDA events, median |
| Git commit | dfac770-dirty |
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

## References

- Milakov & Gimelshein, *Online normalizer calculation for softmax*, [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- Dao et al., *FlashAttention*, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- NVIDIA Nsight Compute CLI documentation — metric names change between versions;
  `scripts/phase0_check.sh` verifies them against `ncu --query-metrics` on the actual
  machine rather than trusting a copied command line.
