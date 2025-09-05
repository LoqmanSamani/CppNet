#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <Eigen/Dense>

namespace CppNet {
    // CPU implementations
    void matmul_cpu(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::MatrixXf &C);
    void matmul_cpu_eigen(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::MatrixXf &C);
    
    #ifdef USE_CUDA
    // GPU implementations  
    void matmul_gpu(const float *A, const float *B, float *C, int M, int N, int K);
    #endif
}

#endif // KERNELS_HPP