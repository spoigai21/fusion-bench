# Runbook — the GPU session, start to finish

Everything in this repo runs without a GPU except the parts that need one. This is the
order to run those parts on a rented box, and what to check at each step. Budget: well
under an hour including container setup; the measurement run itself is under ten minutes.

## Before you rent

Nothing here needs a GPU:

```bash
make sim                               # simulate the reduction logic against FP64
bash scripts/nvcc_check.sh             # compile kernels.cu with real nvcc (needs Docker)
bash scripts/build_check.sh            # compile + link + import the extension for real
python bench/bytes_model.py            # the predictions, printed
python bench/parse_ncu.py --analytic   # what Plan B's table would look like
```

`make sim` reproduces the block structure, the identity elements for idle lanes and the
shuffle order in NumPy. It will not tell you anything about speed, but it catches the
class of bug that is most annoying to debug at $1.39/hr: a reduction that returns NaN
because two idle lanes combined, or a `float4` tail that drops elements.

Read `docs/notes.md` and make sure you still agree with the predictions. After the first
counter is read, they stop being predictions.

## 1. Rent and start the container

RunPod, Lambda or vast.ai. **Start from a PyTorch CUDA image** so the toolkit is present
— building CUDA + PyTorch from scratch is where evenings disappear for reasons unrelated
to CUDA.

Nsight Compute needs counter access, which most containers deny by default. On RunPod,
enable privileged mode / extended permissions when creating the pod. With plain Docker:

```bash
docker run --gpus all --cap-add=SYS_ADMIN -it <pytorch-cuda-image>
```

Cost anchors: A100 80GB PCIe ≈ $1.39/hr on RunPod, billed per second. An RTX 4090 at
≈ $0.34/hr is fine for development — only the final numbers need the A100, and the
README must then say which card produced them.

## 2. Phase 0 — the checkpoint that can kill the project

```bash
git clone <this repo> && cd fusion-bench
pip install -r requirements.txt     # torch is usually already in the image
bash scripts/phase0_check.sh
```

This runs, in order: `nvidia-smi`, `nvcc --version`, `ncu --version`, a torch device
check, **an actual `ncu` profile of `torch.softmax` to confirm a byte count comes back**,
and a `--query-metrics` check that the three metric names exist on this Nsight version.

**Do not proceed until step 5 prints a byte count.** If it fails with a permissions error
(`ERR_NVGPUCTRPERM`), in order: restart the container with `--cap-add=SYS_ADMIN`, try a
different provider, try bare metal. If no machine you can get will give you counters,
switch to Plan B (`make analytic`) and say so plainly in the README — an honest
limitation costs far less than a vague one.

Optional, if you have root:

```bash
sudo bash scripts/lock_clocks.sh        # and 'reset' before releasing the box
```

If you cannot lock clocks, carry on. The harness records achieved SM clock on every run
and the median over 200 launches is robust to the occasional boost change.

## 3. Phase 1 — plumbing

```bash
make build      # python -m bench.ext
```

First run takes ~30–60 s of nvcc; after that it is cached in `.build/`. It compiles the
extension, runs the copy kernel, and prints which v2 path each benchmark shape takes and
how much shared memory v1 needs.

The compile and link half of this has already been verified off-box by
`scripts/build_check.sh`, so if it fails here the cause is almost certainly the
environment — a CUDA/torch version mismatch or a missing toolkit — rather than the
source. If this hangs or errors, nothing downstream is worth
debugging yet.

## 4. Phase 2–5 — correctness

```bash
make verify
```

Every kernel against a float64 NumPy reference at every edge case: `D = 1`, `D % 32 ≠ 0`,
`D % 4 ≠ 0`, a row containing `−inf`, `x ± 1000`, a wide-dynamic-range ramp, and a shape
long enough to force v2's generic path. Pass condition is max absolute error below `1e-5`.

**A kernel that fails here does not get timed.** `run_all.py` enforces that too: a failed
validation writes a `FAIL` row with no timing rather than a number that would invalidate
the whole comparison.

## 5. Phase 6 — the numbers

```bash
make bench                     # ~2 min: 5 functions × 2 shapes × 250 launches
make bench-variants            # add v2a/v2b/v2c to that run (phase 5 attribution)
```

Writes `results/summary.csv`, `results/env.json`, and one file per measurement in
`results/raw/` holding all 200 individual samples. Check p5/p95 against the median before
trusting anything: a spread wider than a few percent usually means the clocks are moving
or something else is on the GPU.

```bash
make profile                   # ncu, one launch per kernel per shape
make profile-variants          # same, for the v2a/v2b/v2c attribution
make bytes                     # -> results/bytes.csv, and prints the table
```

The attribution counters go in a **separate** ncu run, writing
`results/raw/ncu_variants_<N>x<D>.csv`. That is not tidiness: `v2` dispatches to the
same CUDA kernels as `v2b` and `v2c`, so profiling them together leaves rows that cannot
be told apart by kernel name, and their counters would be summed into one wrong row.
`profile_one.py` refuses the ambiguous combination rather than producing it, and the
`ncu_variants_` prefix is what tells `parse_ncu.py` which mapping to apply.

`scripts/profile.sh` points `ncu` at `bench/profile_one.py`, which issues exactly one
launch per kernel. Never point `ncu` at the benchmark loop: it instruments every launch,
so 200 iterations means 200 rows and a very slow run.

```bash
make plot                      # -> results/plots/time.png and time_dark.png
```

## 6. Phase 7 — the write-up

```bash
make tables        # rewrites every table in README.md from results/
```

`bench/make_tables.py` splices the timings, speedups, byte tables and environment block
into the regions marked `<!-- BEGIN:... -->` in README.md, so no number in the write-up
is ever typed by hand. `make check-tables` exits non-zero if the README has drifted from
`results/`, which is the thing to run before committing the final numbers. Then fill in the measurement and
verdict rows in `docs/notes.md` — including the ones where the prediction was wrong.

The write-up needs four things: the chart, the byte table, one sentence tying the speedup
to the traffic reduction (or explaining why it does not hold at 4096×1024), and an honest
conclusion. If `torch.softmax` beats v2, say so and explain the gap from the counters.
*"ATen stayed 1.4× ahead; its persistent kernel avoids a reload my version pays for"* is a
stronger line than any speedup claim, because it shows you can read a profiler and reason
about a kernel you did not write.

## 7. Before you kill the pod

```bash
sudo bash scripts/lock_clocks.sh reset
```

Copy out `results/` — `summary.csv`, `bytes.csv`, `env.json`, `raw/`, `plots/`. Everything
else is regenerable; those files are the deliverable.

## Things that waste an evening

| symptom | cause |
|---|---|
| kernel returns zeros, looks like a correctness bug | a launch failure that was never checked. Every launcher here returns `cudaGetLastError()` and `bindings.cpp` turns it into a `TORCH_CHECK` — keep it that way |
| `ncu` run takes minutes and emits hundreds of rows | it was pointed at the benchmark loop instead of `profile_one.py` |
| `ncu` CSV is unparseable | the profiled script printed to stdout. `profile_one.py` logs to stderr for exactly this reason |
| one timing sample is 50× the rest | `torch.compile` recompiled inside the timed loop. `run_all.py` resets the compile cache per shape so compilation always lands in warmup |
| metric not found | Nsight metric names change between versions; check `ncu --query-metrics` rather than a copied command line |
| `C++20 or later compatible compiler is required` | torch >= 2.6 needs gcc 10+. Every PyTorch CUDA image has it, but a bare CUDA image plus a pip torch may not |
| v1 errors at large `D` | it needs `D·4` bytes of shared memory per block. That ceiling is the point of v2, and `bindings.cpp` reports it as a limit rather than launching something that cannot fit |
