#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <functional>
#include "kernels.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)
#endif

inline double elapsed_ms(std::function<void()> fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    const int M = 1024, K = 1024, N = 1024;
    std::cout << "Matrix size: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    
    // Initialize matrices
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);
    Eigen::MatrixXf C_cpu, C_eigen, C_gpu;
    
    // CPU Benchmark (naive implementation)
    double cpu_time = elapsed_ms([&]() {
        CppNet::matmul_cpu(A, B, C_cpu);
    });
    std::cout << "CPU (naive) time: " << cpu_time << " ms" << std::endl;
    
    // CPU Benchmark (Eigen optimized)
    double eigen_time = elapsed_ms([&]() {
        CppNet::matmul_cpu_eigen(A, B, C_eigen);
    });
    std::cout << "CPU (Eigen) time: " << eigen_time << " ms" << std::endl;
    
    #ifdef USE_CUDA
    // GPU Benchmark
    float *d_A, *d_B, *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    double gpu_time = elapsed_ms([&]() {
        CppNet::matmul_gpu(d_A, d_B, d_C, M, N, K);
    });
    
    C_gpu.resize(M, N);
    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    
    // Correctness checks
    float cpu_gpu_diff = (C_cpu - C_gpu).norm();
    float eigen_gpu_diff = (C_eigen - C_gpu).norm();
    
    std::cout << "CPU vs GPU difference: " << cpu_gpu_diff << std::endl;
    std::cout << "Eigen vs GPU difference: " << eigen_gpu_diff << std::endl;
    std::cout << "Speedup (CPU/GPU): " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << "Speedup (Eigen/GPU): " << eigen_time / gpu_time << "x" << std::endl;
    
    #else
    std::cout << "CUDA not enabled, skipping GPU test." << std::endl;
    #endif
    
    return 0;
}