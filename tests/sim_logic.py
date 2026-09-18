#!/usr/bin/env python3
"""CPU simulation of the three kernels' *logic* -- no GPU, no CUDA toolkit required.

It reproduces the structure that actually goes wrong in a reduction kernel: the same
256-thread block, the same per-thread element mapping, the same identity elements for
idle lanes, the same shared-memory tree and __shfl_down_sync order, in float32.

It does not simulate timing, coalescing or occupancy -- only whether the arithmetic
comes out right. That is enough to catch the bugs that are expensive to find on a rented
box: NaN from combining two identity elements, D=1 with 255 idle threads, the float4
tail, and -inf rows.

    python tests/sim_logic.py        # exits non-zero if any case fails

Run this before renting a GPU. tests/verify.py is the real check, on the real kernels.
"""
import numpy as np

BLOCK = 256
WARP = 32
WARPS = BLOCK // WARP
f32 = np.float32
NEG_INF = f32(-np.inf)


def exp_shift(a, m):
    a = f32(a); m = f32(m)
    return f32(1.0) if a == m else f32(np.exp(np.float64(a) - np.float64(m)))


def online_update(m, d, x):
    mn = f32(max(m, x)) if not (np.isnan(m) or np.isnan(x)) else f32(np.fmax(m, x))
    d = f32(f32(d) * exp_shift(m, mn) + exp_shift(x, mn))
    return mn, d


def online_combine(m, d, m2, d2):
    mn = f32(np.fmax(m, m2))
    d = f32(f32(d) * exp_shift(m, mn) + f32(d2) * exp_shift(m2, mn))
    return mn, d


def warp_reduce_online(ms, ds, width):
    """ms/ds are length-32 arrays (one warp). Mirrors __shfl_down_sync."""
    ms, ds = list(ms), list(ds)
    off = width // 2
    while off > 0:
        nm, nd = list(ms), list(ds)
        for lane in range(WARP):
            src = lane + off
            m2 = ms[src] if src < WARP else ms[lane]   # shfl_down: out-of-range keeps own
            d2 = ds[src] if src < WARP else ds[lane]
            nm[lane], nd[lane] = online_combine(ms[lane], ds[lane], m2, d2)
        ms, ds = nm, nd
        off >>= 1
    return ms, ds


def block_reduce_online(ms, ds):
    """ms/ds length BLOCK. Returns broadcast (m, d)."""
    sm = [None] * WARPS
    sd = [None] * WARPS
    for w in range(WARPS):
        lo, hi = w * WARP, (w + 1) * WARP
        rm, rd = warp_reduce_online(ms[lo:hi], ds[lo:hi], WARP)
        sm[w], sd[w] = rm[0], rd[0]
    # warp 0 reduces the leaders; lanes >= WARPS carry the identity
    lm = [sm[l] if l < WARPS else NEG_INF for l in range(WARP)]
    ld = [sd[l] if l < WARPS else f32(0.0) for l in range(WARP)]
    rm, rd = warp_reduce_online(lm, ld, WARPS)
    return rm[0], rd[0]


def v0_row(row):
    D = len(row)
    # pass 1: per-thread max, then shared-mem tree
    part = [NEG_INF] * BLOCK
    for t in range(BLOCK):
        for i in range(t, D, BLOCK):
            part[t] = f32(np.fmax(part[t], row[i]))
    red = list(part); s = BLOCK // 2
    while s > 0:
        for t in range(s):
            red[t] = f32(np.fmax(red[t], red[t + s]))
        s >>= 1
    m = red[0]
    # pass 2: sum of exp
    part = [f32(0.0)] * BLOCK
    for t in range(BLOCK):
        for i in range(t, D, BLOCK):
            part[t] = f32(part[t] + exp_shift(row[i], m))
    red = list(part); s = BLOCK // 2
    while s > 0:
        for t in range(s):
            red[t] = f32(red[t] + red[t + s])
        s >>= 1
    d = red[0]
    inv = f32(f32(1.0) / d)
    return np.array([f32(exp_shift(v, m) * inv) for v in row], dtype=f32)


v1_row = v0_row  # identical arithmetic; v1 only changes where the row is read from


def v2_vec_row(row):
    """Register-resident float4 path. D % 4 == 0 required."""
    D = len(row)
    Dv = D // 4
    vpt = 1
    while vpt < (Dv + BLOCK - 1) // BLOCK:
        vpt <<= 1
    assert vpt <= 16, "would fall back to generic"
    ms = [NEG_INF] * BLOCK
    ds = [f32(0.0)] * BLOCK
    for t in range(BLOCK):
        for i in range(vpt):
            idx = t + i * BLOCK
            if idx < Dv:
                for k in range(4):
                    ms[t], ds[t] = online_update(ms[t], ds[t], row[idx * 4 + k])
    m, d = block_reduce_online(ms, ds)
    inv = f32(f32(1.0) / d)
    return np.array([f32(exp_shift(v, m) * inv) for v in row], dtype=f32)


def v2_gen_row(row):
    D = len(row)
    ms = [NEG_INF] * BLOCK
    ds = [f32(0.0)] * BLOCK
    for t in range(BLOCK):
        for i in range(t, D, BLOCK):
            ms[t], ds[t] = online_update(ms[t], ds[t], row[i])
    m, d = block_reduce_online(ms, ds)
    inv = f32(f32(1.0) / d)
    return np.array([f32(exp_shift(v, m) * inv) for v in row], dtype=f32)


def reference(row):
    a = np.asarray(row, dtype=np.float64)
    m = a.max()
    e = np.exp(a - m)
    return e / e.sum()


def check(name, row, fn, tol=1e-5):
    got = fn(np.asarray(row, dtype=f32))
    ref = reference(row)
    if not np.isfinite(got).all():
        return f"{name}: NON-FINITE output ({np.count_nonzero(~np.isfinite(got))} values)"
    err = np.abs(got.astype(np.float64) - ref).max()
    status = "ok " if err <= tol else "FAIL"
    return f"{status} {name}: max abs err {err:.3e}"


rng = np.random.default_rng(0)
cases = {
    "D=1": rng.standard_normal(1),
    "D=3": rng.standard_normal(3),
    "D=31": rng.standard_normal(31),
    "D=33": rng.standard_normal(33),
    "D=255": rng.standard_normal(255),
    "D=257": rng.standard_normal(257),
    "D=1023": rng.standard_normal(1023),
    "D=1024": rng.standard_normal(1024),
    "D=2048": rng.standard_normal(2048),
    "D=1024 +1000": rng.standard_normal(1024) + 1000.0,
    "D=1024 -1000": rng.standard_normal(1024) - 1000.0,
    "D=1024 ramp": np.linspace(-80, 80, 1024),
    "D=1024 descending": np.linspace(80, -80, 1024),
}
masked = rng.standard_normal(128); masked[40:] = -np.inf
cases["D=128 masked"] = masked
one_live = rng.standard_normal(1024); one_live[1:] = -np.inf
cases["D=1024 one live"] = one_live

failures = []


def report(line):
    print("  " + line)
    if line.startswith("FAIL") or "NON-FINITE" in line:
        failures.append(line)


print("=== v0 / v1 (shared-memory tree) ===")
for n, r in cases.items():
    report(check(n, r, v0_row))

print("\n=== v2 vec (register-resident float4, D%4==0 only) ===")
for n, r in cases.items():
    if len(r) % 4 == 0:
        report(check(n, r, v2_vec_row))

print("\n=== v2 generic (scalar fallback) ===")
for n, r in cases.items():
    report(check(n, r, v2_gen_row))

print("\n=== fully masked row (undefined input, documented) ===")
allneg = np.full(64, -np.inf)
for nm, fn in (("v0", v0_row), ("v2gen", v2_gen_row)):
    out = fn(np.asarray(allneg, dtype=f32))
    print(f"  {nm}: unique outputs {np.unique(out)}  (expected uniform 1/64 = {1 / 64:.6f})")
print("  torch.softmax returns NaN here; both are defensible for undefined input.")

print()
if failures:
    print(f"{len(failures)} simulated case(s) failed:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("all simulated cases within 1e-5 of the FP64 reference")
