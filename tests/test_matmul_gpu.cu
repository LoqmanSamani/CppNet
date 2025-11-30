#include <iostream>
#include "/home/loqman/Downloads/cproj/CppNet/include/kernels/gpu.hpp"

int main() {
    int M = 2, N = 2, K = 2;

    float A[4] = {1,2,3,4};
    float B[4] = {5,6,7,8};
    float C[4] = {};

    dl_gpu::matmul_gpu(A, B, C, M, N, K);

    std::cout << C[0] << " " << C[1] << std::endl;
    std::cout << C[2] << " " << C[3] << std::endl;

    return 0;
}

