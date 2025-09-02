#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <functional>
#include "kernels.h"   // Declare matmul_cpu / matmul_gpu


#ifdef USE_CUDA
#include <cuda_runtime.h>
extern void matmul_gpu(const float*, const float*, float*, int, int, int);
#endif

inline double elapsed_ms(std::function<void()> fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

int main() {
    int M = 1024, K = 1024, N = 1024;
    std::cout << "Matrix size: " << M << "x" << K << " * " << K << "x" << N << std::endl;

    Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);
    Eigen::MatrixXf C_cpu(M, N);
    Eigen::MatrixXf C_gpu(M, N);

    // -------------------------
    // CPU Benchmark
    // -------------------------
    double cpu_time = elapsed_ms([&](){
        matmul_cpu(A, B, C_cpu);
    });
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;

#ifdef USE_CUDA
    // -------------------------
    // GPU Benchmark
    // -------------------------
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);

    double gpu_time = elapsed_ms([&](){
        matmul_gpu(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize(); // ensure timing is correct
    });

    cudaMemcpy(C_gpu.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;

    // -------------------------
    // Correctness check
    // -------------------------
    float diff = (C_cpu - C_gpu).norm();
    std::cout << "CPU vs GPU difference (Frobenius norm): " << diff << std::endl;
#else
    std::cout << "CUDA not enabled, skipping GPU test." << std::endl;
#endif

    return 0;
}

