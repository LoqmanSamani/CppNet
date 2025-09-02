#ifndef KERNELS_H
#define KERNELS_H
#include <Eigen/Dense>

void matmul_cpu(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::MatrixXf &C);

#ifdef USE_CUDA
void matmul_gpu(const float *A, const float *B, float *C, int M, int N, int K);
#endif

#endif
