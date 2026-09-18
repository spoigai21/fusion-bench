// fusion-bench — launcher declarations shared between bindings.cpp and kernels.cu
//
// kernels.cu is deliberately torch-free: it takes raw pointers and a stream, so the
// kernels can be dropped into another project (or profiled from a C++ driver) without
// dragging libtorch along. All torch-level validation lives in bindings.cpp.
#pragma once

#include <cuda_runtime.h>

// Every launcher returns the result of cudaGetLastError() after the launch, so a
// silently failing kernel surfaces immediately instead of returning zeros.

// Phase 1 plumbing check: y = x, grid-stride copy.
cudaError_t launch_copy(const float* x, float* y, long long n, cudaStream_t stream);

// v0 — naive. One block per row, three separate passes over the row in global memory.
cudaError_t launch_softmax_v0(const float* x, float* y, int N, int D, cudaStream_t stream);

// v1 — fused. Row staged in shared memory once, both reductions done there.
cudaError_t launch_softmax_v1(const float* x, float* y, int N, int D, cudaStream_t stream);

// v2 — online softmax, warp-shuffle reductions, float4 loads, row held in registers.
cudaError_t launch_softmax_v2(const float* x, float* y, int N, int D, cudaStream_t stream);

// Which code path v2 would take for this D — reported by the harness so the CSV says
// whether a row was register-resident or fell back to the generic two-pass kernel.
// Returns "vec<VPT>" or "generic".
const char* softmax_v2_path(const float* x, int D);

// Shared memory (bytes) v1 needs for a row of length D. The harness prints this so the
// occupancy argument in the write-up is backed by a number rather than a guess.
int softmax_v1_smem_bytes(int D);
