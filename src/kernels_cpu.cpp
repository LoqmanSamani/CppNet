#pragma once
#include <Eigen/Dense>

// Naive CPU matmul (triple loop)
inline void matmul_cpu(const Eigen::MatrixXf &A,
                       const Eigen::MatrixXf &B,
                       Eigen::MatrixXf &C) {
    int M = A.rows();
    int K = A.cols();
    int N = B.cols();

    C.setZero(M, N); // important!

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C(i,j) += A(i,k) * B(k,j);
            }
        }
    }
}
