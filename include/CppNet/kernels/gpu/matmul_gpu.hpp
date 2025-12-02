#pragma once

namespace CppNet {
namespace Kernels {
namespace GPU {

void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K);

} // namespace GPU
} // namespace Kernels
} // namespace CppNet
