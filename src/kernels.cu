// fusion-bench — the three-rung softmax ladder.
//
//   v0  naive     : one block per row, three passes over the row in global memory
//   v1  fused     : row staged in shared memory once, both reductions done there
//   v2  online    : single pass (running max + running sum), warp shuffles, float4
//
// Every rung removes one source of memory traffic and nothing else. In particular v0
// already reads coalesced (threads in a block touch consecutive addresses), so the
// v0 -> v1 delta measures fusion alone and is not confounded with access pattern.
//
// Row-length conventions, shared by all three kernels:
//   * exp(x - m) is evaluated as (x == m ? 1 : expf(x - m)). For finite x this is
//     bit-identical to the plain form, but it keeps a fully masked row (every element
//     -inf, so m == -inf) from producing NaN: such a row comes out uniform 1/D.
//     torch.softmax returns NaN there. The case is degenerate either way; it is
//     documented in docs/notes.md and excluded from the FP64 cross-check.
//   * Rows are independent, so the grid is exactly N blocks and no kernel needs a
//     grid-stride loop over rows.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "softmax.h"

namespace {

constexpr int kBlock = 256;  // threads per row-block, power of two (tree reduction)
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = kBlock / kWarp;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr float kNegInf = -INFINITY;

// ---------------------------------------------------------------------------------
// numerics
// ---------------------------------------------------------------------------------

// exp(a - m), with a == m pinned to 1 (see the note at the top of the file).
__device__ __forceinline__ float exp_shift(float a, float m) {
  return (a == m) ? 1.0f : __expf(a - m);
}

// Fold one new element into a running (max, sum-of-exp) pair.
//   m' = max(m, x)
//   d' = d * exp(m - m') + exp(x - m')
// Milakov & Gimelshein, arXiv:1805.02867.
__device__ __forceinline__ void online_update(float& m, float& d, float x) {
  const float mn = fmaxf(m, x);
  d = d * exp_shift(m, mn) + exp_shift(x, mn);
  m = mn;
}

// Combine two running (max, sum) pairs. (-inf, 0) is the identity, including when both
// sides are the identity -- which is why exp_shift's equality case matters here.
__device__ __forceinline__ void online_combine(float& m, float& d, float m2, float d2) {
  const float mn = fmaxf(m, m2);
  d = d * exp_shift(m, mn) + d2 * exp_shift(m2, mn);
  m = mn;
}

// ---------------------------------------------------------------------------------
// block reductions
// ---------------------------------------------------------------------------------

// Shared-memory tree reduction over the block's partials. `red` is kBlock floats.
// blockDim.x is always kBlock (a power of two), so the halving loop is exact.
__device__ __forceinline__ float block_reduce_max(float v, float* red) {
  const int tid = threadIdx.x;
  red[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float r = red[0];
  __syncthreads();  // everyone has read red[0] before the buffer is reused
  return r;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* red) {
  const int tid = threadIdx.x;
  red[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float r = red[0];
  __syncthreads();
  return r;
}

// Shared-memory tree reduction over online (max, sum) pairs. This is what v2a uses:
// the online formulation, but reduced the same way v1 reduces, so that isolating the
// warp-shuffle change in v2b measures the reduction mechanism and nothing else.
// `sm`/`sd` are blockDim floats each.
__device__ __forceinline__ void block_reduce_online_tree(float& m, float& d, float* sm,
                                                         float* sd) {
  const int tid = threadIdx.x;
  sm[tid] = m;
  sd[tid] = d;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      float mm = sm[tid], dd = sd[tid];
      online_combine(mm, dd, sm[tid + s], sd[tid + s]);
      sm[tid] = mm;
      sd[tid] = dd;
    }
    __syncthreads();
  }
  m = sm[0];
  d = sd[0];
  __syncthreads();
}

// Warp-shuffle reduction of an online (max, sum) pair: values move through registers,
// so there is no shared memory and no __syncthreads() inside the warp.
__device__ __forceinline__ void warp_reduce_online(float& m, float& d, int width) {
  for (int off = width >> 1; off > 0; off >>= 1) {
    const float m2 = __shfl_down_sync(kFullMask, m, off);
    const float d2 = __shfl_down_sync(kFullMask, d, off);
    online_combine(m, d, m2, d2);
  }
}

// Block-wide online reduction: reduce within warps through registers, then one small
// shared-memory step across the kWarpsPerBlock warp leaders. Result is broadcast to
// every thread. `sm`/`sd` are kWarpsPerBlock floats each.
__device__ __forceinline__ void block_reduce_online(float& m, float& d, float* sm, float* sd) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x >> 5;

  warp_reduce_online(m, d, kWarp);
  if (lane == 0) {
    sm[warp] = m;
    sd[warp] = d;
  }
  __syncthreads();

  if (warp == 0) {
    // All 32 lanes of warp 0 take part in the shuffles; lanes past the warp count
    // carry the identity element.
    m = (lane < kWarpsPerBlock) ? sm[lane] : kNegInf;
    d = (lane < kWarpsPerBlock) ? sd[lane] : 0.0f;
    warp_reduce_online(m, d, kWarpsPerBlock);
    if (lane == 0) {
      sm[0] = m;
      sd[0] = d;
    }
  }
  __syncthreads();

  m = sm[0];
  d = sd[0];
  __syncthreads();
}

// ---------------------------------------------------------------------------------
// phase 1 plumbing
// ---------------------------------------------------------------------------------

__global__ void copy_kernel(const float* __restrict__ x, float* __restrict__ y, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x) {
    y[i] = x[i];
  }
}

// ---------------------------------------------------------------------------------
// v0 — naive: three passes over the row, every one of them from global memory
// ---------------------------------------------------------------------------------
//
// Traffic model: 3 reads + 1 write = 16*N*D bytes.
// The block reduction uses shared memory, but the *row* never does: each pass streams
// the row out of L2/DRAM again.

__global__ void softmax_v0_kernel(const float* __restrict__ x, float* __restrict__ y, int D) {
  extern __shared__ float s_dyn[];  // kBlock floats
  float* red = s_dyn;

  const long long row = blockIdx.x;
  const float* __restrict__ xr = x + row * (long long)D;
  float* __restrict__ yr = y + row * (long long)D;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  // pass 1 — max
  float local_max = kNegInf;
  for (int i = tid; i < D; i += stride) local_max = fmaxf(local_max, xr[i]);
  const float m = block_reduce_max(local_max, red);

  // pass 2 — sum of exponentials (re-reads the row)
  float local_sum = 0.0f;
  for (int i = tid; i < D; i += stride) local_sum += exp_shift(xr[i], m);
  const float d = block_reduce_sum(local_sum, red);

  // pass 3 — normalize (re-reads the row a third time, recomputing the exp)
  const float inv = 1.0f / d;
  for (int i = tid; i < D; i += stride) yr[i] = exp_shift(xr[i], m) * inv;
}

// ---------------------------------------------------------------------------------
// v1 — fused: the row is read from global memory exactly once
// ---------------------------------------------------------------------------------
//
// Traffic model: 1 read + 1 write = 8*N*D bytes.
// Cost: D*4 bytes of shared memory per block. At D = 8192 that is 32 KB, so only one
// block co-resides per SM on a 48 KB carve-out -- which is exactly what v2 fixes.

__global__ void softmax_v1_kernel(const float* __restrict__ x, float* __restrict__ y, int D) {
  extern __shared__ float s_dyn[];
  float* row_s = s_dyn;      // D floats: the row
  float* red = s_dyn + D;    // kBlock floats: reduction scratch

  const long long row = blockIdx.x;
  const float* __restrict__ xr = x + row * (long long)D;
  float* __restrict__ yr = y + row * (long long)D;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  // the one and only global read of the row
  for (int i = tid; i < D; i += stride) row_s[i] = xr[i];
  __syncthreads();

  float local_max = kNegInf;
  for (int i = tid; i < D; i += stride) local_max = fmaxf(local_max, row_s[i]);
  const float m = block_reduce_max(local_max, red);

  // exponentiate in place, so pass 3 does not have to recompute it
  float local_sum = 0.0f;
  for (int i = tid; i < D; i += stride) {
    const float e = exp_shift(row_s[i], m);
    row_s[i] = e;
    local_sum += e;
  }
  __syncthreads();
  const float d = block_reduce_sum(local_sum, red);

  const float inv = 1.0f / d;
  for (int i = tid; i < D; i += stride) yr[i] = row_s[i] * inv;
}

// ---------------------------------------------------------------------------------
// v2a — online softmax ONLY (attribution rung 1 of 3)
// ---------------------------------------------------------------------------------
//
// One pass to compute the running max and running sum together, where v1 needed two
// passes over shared memory. Everything else is deliberately held at v1's technology:
// scalar loads, shared-memory tree reduction.
//
// Traffic model: 2 reads + 1 write = 12*N*D bytes. Note this is MORE than v1, not less:
// v1 read the row once into shared memory and normalized from there, while v2a holds no
// row at all and must re-read it. The online formulation removes a pass over *shared*
// memory, not over DRAM. If v2a loses to v1, that is the reason, and it is the whole
// point of measuring the rungs separately.

__global__ void softmax_v2a_kernel(const float* __restrict__ x, float* __restrict__ y, int D) {
  extern __shared__ float s_dyn[];  // 2 * kBlock floats
  float* sm = s_dyn;
  float* sd = s_dyn + blockDim.x;

  const long long row = blockIdx.x;
  const float* __restrict__ xr = x + row * (long long)D;
  float* __restrict__ yr = y + row * (long long)D;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  float m = kNegInf;
  float d = 0.0f;
  for (int i = tid; i < D; i += stride) online_update(m, d, xr[i]);

  block_reduce_online_tree(m, d, sm, sd);
  const float inv = 1.0f / d;

  for (int i = tid; i < D; i += stride) yr[i] = exp_shift(xr[i], m) * inv;
}

// ---------------------------------------------------------------------------------
// v2c — online + warp shuffles + float4 + registers (attribution rung 3 of 3)
// ---------------------------------------------------------------------------------
//
// Traffic model: 1 read + 1 write = 8*N*D bytes, same as v1, but with no shared memory
// holding the row, so many blocks co-reside per SM at large D. This is also the kernel
// `softmax_v2` dispatches to whenever the shape allows it.
//
// VPT = float4s per thread. The row lives in `reg`, so the normalize pass needs neither
// a global re-read (which the generic fallback below pays for) nor shared memory.

template <int VPT>
__global__ void softmax_v2_vec_kernel(const float4* __restrict__ x, float4* __restrict__ y,
                                      int Dv) {
  __shared__ float sm[kWarpsPerBlock];
  __shared__ float sd[kWarpsPerBlock];

  const long long row = blockIdx.x;
  const float4* __restrict__ xr = x + row * (long long)Dv;
  float4* __restrict__ yr = y + row * (long long)Dv;
  const int tid = threadIdx.x;

  float4 reg[VPT];
  float m = kNegInf;
  float d = 0.0f;

  // single pass: 128-bit loads, running max and running sum maintained together
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int idx = tid + i * kBlock;
    if (idx < Dv) {
      const float4 v = xr[idx];
      reg[i] = v;
      online_update(m, d, v.x);
      online_update(m, d, v.y);
      online_update(m, d, v.z);
      online_update(m, d, v.w);
    } else {
      reg[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }

  block_reduce_online(m, d, sm, sd);
  const float inv = 1.0f / d;

#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int idx = tid + i * kBlock;
    if (idx < Dv) {
      const float4 v = reg[i];
      float4 o;
      o.x = exp_shift(v.x, m) * inv;
      o.y = exp_shift(v.y, m) * inv;
      o.z = exp_shift(v.z, m) * inv;
      o.w = exp_shift(v.w, m) * inv;
      yr[idx] = o;
    }
  }
}

// v2b — online + warp shuffles (attribution rung 2 of 3), and v2's generic fallback for
// any D or alignment. Identical to v2a except that the block reduction runs through
// registers via __shfl_down_sync instead of a shared-memory tree, so the v2a -> v2b
// delta isolates the reduction mechanism.
// Traffic model: 2 reads + 1 write = 12*N*D bytes. The harness records which path ran.
__global__ void softmax_v2_gen_kernel(const float* __restrict__ x, float* __restrict__ y, int D) {
  __shared__ float sm[kWarpsPerBlock];
  __shared__ float sd[kWarpsPerBlock];

  const long long row = blockIdx.x;
  const float* __restrict__ xr = x + row * (long long)D;
  float* __restrict__ yr = y + row * (long long)D;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  float m = kNegInf;
  float d = 0.0f;
  for (int i = tid; i < D; i += stride) online_update(m, d, xr[i]);

  block_reduce_online(m, d, sm, sd);
  const float inv = 1.0f / d;

  for (int i = tid; i < D; i += stride) yr[i] = exp_shift(xr[i], m) * inv;
}

// ---------------------------------------------------------------------------------
// dispatch helpers
// ---------------------------------------------------------------------------------

// v2's vector path needs the row base pointer 16B-aligned and the row length a multiple
// of 4. Rows start at row * D * 4 bytes from the base, so D % 4 == 0 makes every row
// aligned if the base is.
inline bool v2_vec_eligible(const void* p, int D) {
  return (D % 4 == 0) && ((uintptr_t)p % 16 == 0);
}

// VPT rounded up to a power of two, or 0 if the row is too long to hold in registers.
inline int v2_vpt_for(int D) {
  const int Dv = D / 4;
  const int need = (Dv + kBlock - 1) / kBlock;
  for (int vpt = 1; vpt <= 16; vpt <<= 1) {
    if (need <= vpt) return vpt;
  }
  return 0;  // > 16 float4s/thread (D > 16384): registers would spill, use the fallback
}

}  // namespace

// ---------------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------------

cudaError_t launch_copy(const float* x, float* y, long long n, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  const int block = kBlock;
  long long grid = (n + block - 1) / block;
  if (grid > 65535) grid = 65535;  // grid-stride loop covers the rest
  copy_kernel<<<(int)grid, block, 0, stream>>>(x, y, n);
  return cudaGetLastError();
}

cudaError_t launch_softmax_v0(const float* x, float* y, int N, int D, cudaStream_t stream) {
  if (N == 0 || D == 0) return cudaSuccess;
  const size_t smem = (size_t)kBlock * sizeof(float);
  softmax_v0_kernel<<<N, kBlock, smem, stream>>>(x, y, D);
  return cudaGetLastError();
}

int softmax_v1_smem_bytes(int D) {
  return (int)(((size_t)D + kBlock) * sizeof(float));
}

cudaError_t launch_softmax_v1(const float* x, float* y, int N, int D, cudaStream_t stream) {
  if (N == 0 || D == 0) return cudaSuccess;
  const size_t smem = ((size_t)D + kBlock) * sizeof(float);

  // Anything above the 48 KB default carve-out has to be opted into explicitly. The
  // opt-in is sticky and monotonic, so it is done once per largest-D-seen rather than
  // on every launch -- a cudaFuncSetAttribute call inside a timed loop would show up
  // as kernel time that is not kernel time.
  // (Single-device assumption: the attribute is per-device, and this project
  // benchmarks one GPU. A multi-GPU user would need to key this by device.)
  static const size_t kDefaultCap = 48u * 1024u;
  static int s_configured = 0;
  if (smem > kDefaultCap && (int)smem > s_configured) {
    cudaError_t err = cudaFuncSetAttribute(softmax_v1_kernel,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int)smem);
    if (err != cudaSuccess) return err;  // D too large for this device's shared memory
    s_configured = (int)smem;
  }

  softmax_v1_kernel<<<N, kBlock, smem, stream>>>(x, y, D);
  return cudaGetLastError();
}

cudaError_t launch_softmax_v2(const float* x, float* y, int N, int D, cudaStream_t stream) {
  if (N == 0 || D == 0) return cudaSuccess;

  if (v2_vec_eligible(x, D) && v2_vec_eligible(y, D)) {
    const int Dv = D / 4;
    const auto* xv = reinterpret_cast<const float4*>(x);
    auto* yv = reinterpret_cast<float4*>(y);
    switch (v2_vpt_for(D)) {
      case 1: softmax_v2_vec_kernel<1><<<N, kBlock, 0, stream>>>(xv, yv, Dv); return cudaGetLastError();
      case 2: softmax_v2_vec_kernel<2><<<N, kBlock, 0, stream>>>(xv, yv, Dv); return cudaGetLastError();
      case 4: softmax_v2_vec_kernel<4><<<N, kBlock, 0, stream>>>(xv, yv, Dv); return cudaGetLastError();
      case 8: softmax_v2_vec_kernel<8><<<N, kBlock, 0, stream>>>(xv, yv, Dv); return cudaGetLastError();
      case 16: softmax_v2_vec_kernel<16><<<N, kBlock, 0, stream>>>(xv, yv, Dv); return cudaGetLastError();
      default: break;  // fall through to the generic path
    }
  }

  softmax_v2_gen_kernel<<<N, kBlock, 0, stream>>>(x, y, D);
  return cudaGetLastError();
}

// --- attribution variants -------------------------------------------------------
// v2 above is the rung that goes in the write-up; these three exist so the win can be
// split across the changes that produced it. v2c is what v2 dispatches to when the
// shape allows, and v2b is its generic fallback, so measuring all three costs nothing
// in extra kernel code -- only v2a is unique to this breakdown.

cudaError_t launch_softmax_v2a(const float* x, float* y, int N, int D, cudaStream_t stream) {
  if (N == 0 || D == 0) return cudaSuccess;
  const size_t smem = 2u * (size_t)kBlock * sizeof(float);
  softmax_v2a_kernel<<<N, kBlock, smem, stream>>>(x, y, D);
  return cudaGetLastError();
}

cudaError_t launch_softmax_v2b(const float* x, float* y, int N, int D, cudaStream_t stream) {
  if (N == 0 || D == 0) return cudaSuccess;
  softmax_v2_gen_kernel<<<N, kBlock, 0, stream>>>(x, y, D);
  return cudaGetLastError();
}

cudaError_t launch_softmax_v2c(const float* x, float* y, int N, int D, cudaStream_t stream) {
  if (N == 0 || D == 0) return cudaSuccess;
  // No silent fallback here. If v2c cannot run this shape the caller must be told,
  // because a v2c row that quietly measured v2b would corrupt the attribution.
  if (!softmax_v2c_eligible(x, D) || !softmax_v2c_eligible(y, D)) return cudaErrorInvalidValue;

  const int Dv = D / 4;
  const auto* xv = reinterpret_cast<const float4*>(x);
  auto* yv = reinterpret_cast<float4*>(y);
  switch (v2_vpt_for(D)) {
    case 1: softmax_v2_vec_kernel<1><<<N, kBlock, 0, stream>>>(xv, yv, Dv); break;
    case 2: softmax_v2_vec_kernel<2><<<N, kBlock, 0, stream>>>(xv, yv, Dv); break;
    case 4: softmax_v2_vec_kernel<4><<<N, kBlock, 0, stream>>>(xv, yv, Dv); break;
    case 8: softmax_v2_vec_kernel<8><<<N, kBlock, 0, stream>>>(xv, yv, Dv); break;
    case 16: softmax_v2_vec_kernel<16><<<N, kBlock, 0, stream>>>(xv, yv, Dv); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

bool softmax_v2c_eligible(const float* x, int D) {
  return v2_vec_eligible(x, D) && v2_vpt_for(D) != 0;
}

const char* softmax_v2_path(const float* x, int D) {
  if (v2_vec_eligible(x, D)) {
    switch (v2_vpt_for(D)) {
      case 1: return "vec1";
      case 2: return "vec2";
      case 4: return "vec4";
      case 8: return "vec8";
      case 16: return "vec16";
      default: break;
    }
  }
  return "generic";
}
