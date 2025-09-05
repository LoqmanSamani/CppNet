#include "kernels.hpp"
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet {
    void matmul_cpu(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::MatrixXf &C) {
        int M = A.rows();
        int K = A.cols();
        int N = B.cols();
        
        // Resize output matrix
        C.resize(M, N);
        C.setZero();
        
        // OpenMP parallelized version
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < N; j++) {
                    C(i, j) += A(i, k) * B(k, j);
                }
            }
        }
    }
    
    // Alternative: Use Eigen's optimized multiplication
    void matmul_cpu_eigen(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::MatrixXf &C) {
        C.noalias() = A * B;  // Eigen's highly optimized BLAS implementation
    }
}
