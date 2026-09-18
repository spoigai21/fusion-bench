// fusion-bench — torch extension entry points.
//
// Everything here is validation and plumbing: shape/dtype/contiguity checks, output
// allocation on the right device and stream, and a hard error on any launch failure.
// A kernel that fails silently and returns zeros looks exactly like a correctness bug,
// so every launch is checked before the tensor goes back to Python.

#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <limits>
#include <string>

#include "softmax.h"

namespace {

void check_input(const torch::Tensor& x, const char* who) {
  TORCH_CHECK(x.is_cuda(), who, ": input must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), who, ": input must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, who, ": input must be float32, got ",
              x.scalar_type());
}

void check_2d(const torch::Tensor& x, const char* who) {
  check_input(x, who);
  TORCH_CHECK(x.dim() == 2, who, ": input must be 2D (N, D), got ", x.dim(), "D");
  TORCH_CHECK(x.size(1) > 0, who, ": D must be > 0");
  TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max(), who, ": D too large");
}

void check_launch(cudaError_t err, const char* who) {
  TORCH_CHECK(err == cudaSuccess, who, ": CUDA launch failed: ", cudaGetErrorString(err));
}

using Launcher = cudaError_t (*)(const float*, float*, int, int, cudaStream_t);

torch::Tensor run_softmax(const torch::Tensor& x, Launcher launcher, const char* who) {
  check_2d(x, who);
  const at::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty_like(x);
  const int N = static_cast<int>(x.size(0));
  const int D = static_cast<int>(x.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();
  check_launch(launcher(x.data_ptr<float>(), y.data_ptr<float>(), N, D, stream), who);
  return y;
}

}  // namespace

// Phase 1: the plumbing check. y = x.
torch::Tensor copy_tensor(const torch::Tensor& x) {
  check_input(x, "copy");
  const at::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  check_launch(launch_copy(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), stream), "copy");
  return y;
}

torch::Tensor softmax_v0(const torch::Tensor& x) {
  return run_softmax(x, &launch_softmax_v0, "softmax_v0");
}

torch::Tensor softmax_v1(const torch::Tensor& x) {
  // Shared memory is the binding constraint for v1; fail loudly rather than launching
  // a kernel that cannot fit, so the limitation shows up as an error not a wrong number.
  check_2d(x, "softmax_v1");
  const int D = static_cast<int>(x.size(1));
  int max_smem = 0;
  const int dev = static_cast<int>(x.device().index());
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  TORCH_CHECK(softmax_v1_smem_bytes(D) <= max_smem, "softmax_v1: D=", D, " needs ",
              softmax_v1_smem_bytes(D), " B of shared memory, device max is ", max_smem,
              " B. This is the v1 scaling limit that motivates v2.");
  return run_softmax(x, &launch_softmax_v1, "softmax_v1");
}

torch::Tensor softmax_v2(const torch::Tensor& x) {
  return run_softmax(x, &launch_softmax_v2, "softmax_v2");
}

// Introspection used by the harness so the CSV records which path actually ran.
std::string v2_path(const torch::Tensor& x) {
  check_2d(x, "v2_path");
  return std::string(softmax_v2_path(x.data_ptr<float>(), static_cast<int>(x.size(1))));
}

int64_t v1_smem_bytes(int64_t D) {
  return softmax_v1_smem_bytes(static_cast<int>(D));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "fusion-bench softmax ladder: v0 naive, v1 shared-memory fused, v2 online";
  m.def("copy", &copy_tensor, "identity copy kernel (phase 1 plumbing check)");
  m.def("softmax_v0", &softmax_v0, "v0: naive, three global-memory passes");
  m.def("softmax_v1", &softmax_v1, "v1: fused via shared memory, one global read");
  m.def("softmax_v2", &softmax_v2, "v2: online softmax, warp shuffles, float4");
  m.def("v2_path", &v2_path, "which v2 code path a tensor of this shape takes");
  m.def("v1_smem_bytes", &v1_smem_bytes, "shared memory v1 needs for a row of length D");
}
