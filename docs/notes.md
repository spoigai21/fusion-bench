# Notes — prediction, measurement, verdict per rung

The predictions below were written **before any counter was read**, so they can be wrong
in public. Fill in the measurement and verdict columns after `make bench` and
`make profile`; do not edit a prediction after seeing a number.

Traffic figures come from `bench/bytes_model.py` (FP32, 4 B/element, cache-free model:
every pass over the row costs a full DRAM trip).

---

## The ladder

| | what it does | global passes over the row | modelled bytes |
|---|---|---|---|
| v0 | max, sum, normalize — three separate passes, all from global memory | 3 read + 1 write | `16·N·D` |
| v1 | row staged in shared memory once; both reductions run there | 1 read + 1 write | `8·N·D` |
| v2 | one online pass (running max + running sum), row kept in registers | 1 read + 1 write | `8·N·D` |

Note what this table does *not* say: **v2 does not move fewer bytes than v1.** The
instructions' "reads roughly halve again" refers to passes over the row *inside* the
kernel — v1 makes two of them (max, then sum) against shared memory, v2 makes one — and
those passes never touched DRAM in v1 to begin with. Writing the model down first is
what makes that visible. So:

- **v0 → v1 is a traffic win.** Predicted 2.0× less total traffic (3 reads become 1).
  This is the rung the project's claim rests on.
- **v1 → v2 is an occupancy win, not a traffic win.** v1 needs `D·4` bytes of shared
  memory per block — 32 KB at `D = 8192`, so roughly one block per SM on a 48 KB
  carve-out. v2 holds the row in registers instead, so many blocks co-reside. If v2 beats
  v1 at `D = 8192` while moving the same bytes, that is the evidence, and the DRAM
  counters should show it: equal bytes, less time, higher achieved bandwidth.

A prediction that separates the two is more useful than one that claims both rungs are
the same kind of win.

### On "roughly 3× the ideal traffic" for v0

v0 reads the row 3× where the floor is 1×, but it still writes exactly once. Counting
reads alone gives 3×; counting total DRAM traffic gives (3+1)/(1+1) = **2×**. The byte
table measures total traffic, so 2× is the number to compare the speedup against.

---

## Predictions (written before profiling)

### Rung 1 — v0 → v1

| | |
|---|---|
| Predicted traffic ratio | 2.00× |
| Predicted speedup at 4096×8192 | **< 2×**, because v0's re-reads partly hit L2 even at this size, and because v1's 32 KB/block shared memory holds occupancy to ~1 block/SM |
| Predicted speedup at 4096×1024 | **well under 2×** — the 16.8 MB working set fits inside the A100's 40 MB L2, so v0's extra passes are L2 hits, not DRAM traffic |
| Measured traffic ratio | _(fill in from results/bytes.csv)_ |
| Measured speedup | _(fill in from results/summary.csv)_ |
| Verdict | |

### Rung 2 — v1 → v2

| | |
|---|---|
| Predicted traffic ratio | 1.00× — same bytes |
| Predicted speedup at 4096×8192 | 1.3–1.8×, entirely from occupancy: no shared-memory ceiling, warp-shuffle reductions instead of `__syncthreads()` trees, 128-bit loads |
| Predicted speedup at 4096×1024 | small, maybe none. At `D = 1024` v1 needs only 4 KB/block, so the occupancy argument barely applies |
| Measured traffic ratio | |
| Measured speedup | |
| Verdict | |

### Against `torch.softmax`

Predicted: **ATen wins or ties at 4096×1024**, where its persistent warp-per-row kernel
is exactly the right shape and our v2 has nothing left to exploit. At 4096×8192 the
outcome is genuinely open. If ATen stays ahead, the interesting output of this project is
the explanation from the counters, not a speedup number.

---

## The L2 caveat — why the small shape is expected to break the model

4096×1024 in FP32 is 16.8 MB. The A100's L2 is 40 MB. The **entire input fits in L2**,
so v0's second and third passes over the row are served from L2 rather than DRAM, and the
DRAM counter will show far less than the modelled `3 reads`. The "N× fewer bytes, N×
faster" relationship should therefore *fail* at this shape — and that failure is the most
informative result in the project, provided it is demonstrated rather than asserted.

That is why `lts__t_bytes.sum` is captured alongside the DRAM metrics. The prediction:

- at 4096×1024: `dram_vs_ideal` for v0 ≈ **1.0–1.5×** (not 2×), while `l2_total_bytes`
  for v0 is ≈ 2× that of v1 — the traffic reduction is real, it just happens in L2.
- at 4096×8192 (134 MB, well past L2): `dram_vs_ideal` for v0 ≈ **2×**, and the speedup
  should track it.

---

## Numerics conventions, decided once and applied to all three kernels

- `exp(x - m)` is evaluated as `(x == m) ? 1 : expf(x - m)`. For finite `x` this is
  identical; it exists so that a **fully masked row** (every element `-inf`, so `m` is
  `-inf`) does not produce `NaN` through `exp(-inf - -inf)`. Such a row comes out uniform
  `1/D`. `torch.softmax` returns `NaN` there. Both are defensible for an undefined input;
  the case is excluded from the FP64 cross-check rather than silently differing.
- The online combine treats `(m, d) = (-inf, 0)` as its identity element, which is what
  idle lanes carry when `D` is smaller than the block. Without the equality guard above,
  combining two identity elements yields `NaN`, which is exactly the bug that makes
  `D = 1` fail in a warp-shuffle implementation.
- `--use_fast_math` and `__expf` are used throughout. Output values are bounded by 1, so
  the ~2 ulp intrinsic error is far below the 1e-5 absolute tolerance; the cross-check
  confirms this at every shape rather than assuming it.

## v2's two code paths

| condition | path | traffic |
|---|---|---|
| `D % 4 == 0`, 16 B aligned, `D ≤ 16384` | `vec<VPT>` — float4, row resident in registers | 1 read + 1 write |
| anything else | `generic` — scalar, re-reads the row to normalize | 2 reads + 1 write |

Both benchmark shapes take the register path (`vec1` at `D = 1024`, `vec8` at
`D = 8192`). The generic path exists so the edge cases in `tests/verify.py` are answered
correctly rather than rejected, and every CSV records which path ran so a timing can
never be quietly attributed to the wrong one.

---

## Log

_One line per run: date, GPU, git commit, what changed, what moved._

| date | GPU | commit | change | result |
|---|---|---|---|---|
| | | | | |
