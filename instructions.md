# Fusion Bench

# fusion-bench — Development Instructions

How to build this project from an empty directory to a finished README. Phases are
ordered so that anything capable of killing the project happens on day one, and each
phase ends with a checkpoint you must pass before moving on.

## What this project is

Three softmax kernels forming a ladder — multi-kernel, fused, single-pass — where each
rung removes one source of memory traffic, and Nsight Compute counters show that the
measured speedup equals the measured reduction in DRAM bytes.

The claim the project makes is one sentence: **v1 moves N× fewer bytes than v0 and is
N× faster, so the speedup is the traffic reduction.** Everything below exists to support
or refute that sentence with data.

### Relationship to CUDA-SGEMM-Optimization

That repo already contains a fused softmax kernel benchmarked against torch. This is not
a rewrite of it. The delta:

| Already exists there | What this project adds |
|---|---|
| One final fused kernel | A three-rung ladder, so the win is attributable |
| Two-pass max/sum | Online softmax (single pass, running max) |
| GB/s derived from timing | Measured `dram__bytes_*` from Nsight counters |
| Counters unavailable on that box | Counter access verified before any code is written |
| "~70–75% of HBM peak" asserted | Achieved bandwidth computed from measured traffic |

If counters work here, backporting the byte table into the SGEMM repo's fused section is
worth doing afterwards.

---

## Phase 0 — GPU access and counter access

**Do this before writing a single line of CUDA.** The entire project rests on Nsight
Compute being able to read hardware counters. Many cloud containers and every Colab
runtime block them.

Rent an instance (RunPod, Lambda, or vast.ai; an A100 PCIe is ~$1.39/hr on RunPod and
bills per second). Start from a PyTorch CUDA container image so the toolkit is present.

Verify in this order:

```bash
nvidia-smi                      # GPU visible
nvcc --version                  # compiler present
ncu --version                   # profiler present
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Then the test that actually matters — profile anything at all and confirm counters come
back rather than a permissions error:

```bash
ncu --metrics dram__bytes_read.sum \
    python -c "import torch; x=torch.randn(4096,4096,device='cuda'); torch.softmax(x,-1); torch.cuda.synchronize()"
```

**If this fails with a permissions error**, try in order: `--cap-add=SYS_ADMIN` on the
container, a different provider, or a bare-metal instance. If no machine you can get
will give you counters, do not abandon the project — switch to Plan B below and state it
plainly in the README. An honest limitation costs you far less than a vague one.

> **Plan B (counters blocked):** derive bytes moved analytically from the algorithm —
> v0 makes 3 passes so `3 × N × D × 4` read plus writes, v2 makes 1 — and report the
> predicted ratio against the measured speedup. Weaker evidence, but stated honestly it
> still supports the argument. Say explicitly in the README that these are derived, not
> measured.

**Checkpoint:** `ncu` prints a byte count. Do not proceed without this.

---

## Phase 1 — Build plumbing

Get a trivial kernel compiling and callable from Python before writing real ones.

Use `torch.utils.cpp_extension.load` rather than hand-rolling a build. It compiles on
import, handles the architecture flags, and gives you PyTorch tensors on both sides:

```python
from torch.utils.cpp_extension import load
softmax_ext = load(
    name="softmax_ext",
    sources=["src/bindings.cpp", "src/kernels.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)
```

Write one kernel that copies input to output, call it on a small tensor, and check the
result matches. That is the whole of phase 1.

Two things that waste an evening if you skip them: add `TORCH_CHECK(x.is_cuda())` and
`TORCH_CHECK(x.is_contiguous())` in the binding, and wrap every launch in a
`cudaGetLastError()` check. A silently failing kernel that returns zeros looks exactly
like a correctness bug for an hour before you find it.

**Checkpoint:** `python -c "import bench.ext"` compiles and a copy kernel round-trips.

---

## Phase 2 — Baselines and the correctness harness

Write the harness before the kernels. If timing and validation already work, each kernel
becomes a small, testable unit.

### The four baselines

```python
# 1. Naive composition — genuinely multi-kernel, several round trips
e = (x - x.max(-1, keepdim=True).values).exp()
out = e / e.sum(-1, keepdim=True)

# 2. torch.softmax — ATen's fused kernel. THE REAL BAR.
out = torch.softmax(x, dim=-1)

# 3. torch.compile on the naive composition — automatic fusion
compiled = torch.compile(naive_composition)

# 4. NumPy float64 — the correctness reference, not a speed baseline
```

`torch.softmax` is already fused and for these shapes uses a persistent warp-per-row
kernel. Expect it to be hard to beat. Beating the naive composition is the achievable
result; matching ATen is a good one.

### Correctness

Reference is NumPy in float64, computed from the same fixed-seed input. Report max
absolute error per kernel per shape.

Required test cases:

- `D` not a multiple of 32 — warp remainder handling
- `D` not a multiple of 4 — `float4` tail handling
- `D = 1` — degenerate row
- A row containing `-inf` — masked positions, as in causal attention
- **Large-magnitude input (e.g. `x + 1000`)** — overflows without max subtraction.
  Include this one deliberately: it demonstrates *why* the max subtraction exists, and
  it is the case a naive implementation silently fails.

Pass condition: max absolute error below `1e-5` against the FP64 reference. A kernel
that fails does not get timed.

### Timing protocol

```python
start, end = torch.cuda.Event(True), torch.cuda.Event(True)
for _ in range(50):          # warmup — discard
    fn(x)
torch.cuda.synchronize()
times = []
for _ in range(200):         # timed
    start.record(); fn(x); end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # µs
median = statistics.median(times)
```

CUDA events, 50 warmup, 200 timed, **median** — not best-of-N. Log every individual
timing to `results/raw/` so the distribution is checkable. Log p5 and p95 alongside the
median.

Lock clocks if you have permission, and log achieved clocks either way:

```bash
sudo nvidia-smi --lock-gpu-clocks=1095,1095
sudo nvidia-smi --lock-memory-clocks=1215,1215
```

**Checkpoint:** all four baselines run, validate against FP64, and produce a CSV row.

---

## Phase 3 — v0, naive

One thread block per row. Three separate passes over the row, all in global memory: find
the max, compute the sum of exponentials, then normalize. Each pass re-reads the row.

Deliberately bad, but not *stupidly* bad — threads within the block read consecutive
addresses, so loads are already coalesced. This matters: it means the v0 → v1 delta is a
clean measurement of fusion alone, not fusion confounded with coalescing.

Expect roughly 3× the ideal traffic and a loss to `torch.softmax`.

**Checkpoint:** v0 passes correctness at all test shapes and appears in the CSV.

---

## Phase 4 — v1, fused with shared memory

Load the row into shared memory once. Reduce for max there, reduce for sum there, write
the result once. The row is read from global memory a single time instead of three.

Implementation notes:

- Tree reduction in shared memory, `__syncthreads()` between halving steps
- Shared memory needed is `D × 4` bytes per block. At `D = 8192` that is 32 KB, under
  the 48 KB default cap but high enough to limit how many blocks co-reside per SM.
  Expect v1 to look relatively worse at the large shape for exactly this reason — note
  it, because it motivates v2.

**This is where the big win appears.** It is also the rung the whole project is about.

**Checkpoint:** v1 beats v0 measurably, and you can state the expected byte ratio before
looking at the counters.

---

## Phase 5 — v2, online softmax with warp reductions

Three changes, and if you want clean attribution, land them as separate commits with a
timing run after each.

**Online softmax.** Maintain the running maximum and running sum in a single pass,
rescaling the sum whenever a new maximum appears:

```
m_new = max(m_old, x)
d_new = d_old * exp(m_old - m_new) + exp(x - m_new)
```

This is Milakov & Gimelshein (arXiv:1805.02867), and the same rescaling trick that makes
FlashAttention work. One pass instead of two, so reads roughly halve again.

**Warp-shuffle reduction.** Replace the shared-memory tree with `__shfl_down_sync`,
which exchanges values between threads in a warp through registers. No shared memory, no
`__syncthreads()`. For rows longer than one warp, reduce within warps first, then do a
small shared-memory step across warps.

**Vectorized loads.** Load four floats at a time with `float4` so each thread issues
128-bit transactions. Requires `D % 4 == 0`; handle the remainder with a scalar tail —
and keep the tail path in the correctness tests.

Because v2 never stages the full row in shared memory, it should scale to large `D`
where v1 degrades.

**Checkpoint:** v2 passes correctness including the `-inf` and large-magnitude cases.
Stop here. Three kernels tell the whole story.

---

## Phase 6 — Measurement

### Shapes

Two only: **4096×1024** and **4096×8192**. Skip the full sweep.

> **Expect the small shape to break the model, and say so.** 4096×1024 in FP32 is 16 MB,
> which fits inside the A100's 40 MB L2 cache. v0's extra passes will mostly hit L2
> rather than DRAM, so the DRAM counters will show far less traffic than the algorithm
> implies and the "N× fewer bytes, N× faster" relationship will not hold there. At
> 4096×8192 you are at 128 MB, well past L2, and the model works.
>
> This is the most interesting paragraph in your README, not a problem. Capture
> `lts__t_bytes.sum` (L2 traffic) alongside the DRAM metrics so you can demonstrate it
> rather than speculate.

### The byte table

Profile a **separate single-shot script**, not the benchmark loop — `ncu` instruments
every launch, so pointing it at 200 iterations gives you 200 rows and a very slow run.

```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
    --csv python bench/profile_one.py
```

Three rows, one per kernel. Compute achieved bandwidth as `bytes_moved / time` and
express it as a percentage of the card's stated peak (A100 80GB PCIe: 1935 GB/s; SXM4:
2039 GB/s — state which).

### The chart

One chart: time per version, with `torch.softmax` and `torch.compile` drawn as reference
lines. Save to `results/plots/`.

**Checkpoint:** `results/summary.csv` and `results/bytes.csv` both exist and the README
table can be populated from them without hand-typing a number.

---

## Phase 7 — The write-up

The README is the deliverable. About 400 words plus the tables.

Required:

1. **The chart** — time per version plus the two torch references
2. **The three-row byte table** — the evidence
3. **One sentence tying them together** — the traffic reduction equals the speedup, or
   an explanation of why it does not at the small shape
4. **An honest conclusion**

On that last point: if `torch.softmax` beats v2, say so and explain the gap from the
counters. *"ATen stayed 1.4× ahead; its persistent kernel avoids a reload my version
pays for"* is a stronger line than any speedup claim, because it shows you can read a
profiler and reason about a kernel you did not write.

State the limitations plainly: forward-only, FP32, 2D, last-dim, no autograd. torch's op
is general. The comparison is like-for-like on the forward pass only and is not a claim
of beating PyTorch.

---

## Suggested pace

Six evenings, with the first budgeted at two — container setup is where nights
disappear for reasons unrelated to CUDA.

| Evening | Goal |
|---|---|
| 1 | Phase 0 + 1: GPU access, `ncu` counters confirmed, hello-world kernel compiles |
| 2 | Phase 2: baselines, FP64 correctness harness, timing protocol |
| 3 | Phase 3: v0 |
| 4 | Phase 4: v1 |
| 5 | Phase 5: v2 |
| 6 | Phase 6 + 7: benchmarks, CSVs, chart, byte table, README |

Total GPU time for the final measurement run: under 10 minutes. Develop on a cheap card
(RTX 4090, ~$0.34/hr) and only move to the A100 for the numbers that go in the README.

---

## Repository layout

```
src/
  kernels.cu           v0, v1, v2 — all three kept in the tree
  bindings.cpp         torch extension entry points
bench/
  harness.py           timing protocol — events, warmup, median
  baselines.py         naive composition, torch.softmax, torch.compile
  run_all.py           full sweep -> results/summary.csv
  profile_one.py       single-shot launches, for ncu only
  plot.py              chart -> results/plots/
tests/
  verify.py            FP64 cross-check, all edge-case shapes
results/
  summary.csv          timings, generated
  bytes.csv            ncu counters, generated
  raw/                 every individual timing
  plots/
docs/
  notes.md             prediction, measurement, verdict per rung
```

## Environment to record

Every CSV row carries these, written into the file header. A number without its
environment is not reproducible.

| | |
|---|---|
| GPU | model and memory |
| Peak memory bandwidth | stated spec — the basis for all "% of peak" figures |
| Driver / CUDA runtime | |
| PyTorch version | |
| Clocks | locked, or achieved per iteration |
| Timing | 50 warmup + 200 timed launches, CUDA events, median |
| Git commit | |

## References

- Milakov & Gimelshein, *Online normalizer calculation for softmax*, arXiv:1805.02867
- Dao et al., *FlashAttention*, arXiv:2205.14135
- NVIDIA Nsight Compute CLI documentation — metric names change between versions; verify
  against `ncu --query-metrics` on your machine rather than trusting a copied command