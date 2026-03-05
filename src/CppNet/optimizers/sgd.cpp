#include "CppNet/optimizers/sgd.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Optimizers
    {

        SGD::SGD(const std::string& device, int gpu_block_size)
            : device_(device), gpu_block_size_(gpu_block_size){}

        SGD::~SGD()
        {
#ifdef USE_CUDA
            release_gpu();
#endif
        }

#ifdef USE_CUDA

        void SGD::ensure_gpu(int size)
        {
            if (gpu_buf_ >= size) return;
            release_gpu();
            cudaMalloc(&d_weights_, size * sizeof(float));
            cudaMalloc(&d_gradients_, size * sizeof(float));
            gpu_buf_ = size;
        }

        void SGD::release_gpu()
        {
            if (d_weights_)   { cudaFree(d_weights_);   d_weights_   = nullptr; }
            if (d_gradients_) { cudaFree(d_gradients_); d_gradients_ = nullptr; }
            gpu_buf_ = 0;
        }

        void SGD::update_gpu(float* weights, const float* gradients,
                             int size, float learning_rate)
        {
            ensure_gpu(size);
            cudaMemcpy(d_weights_, weights, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_gradients_, gradients, size * sizeof(float), cudaMemcpyHostToDevice);

            int block = gpu_block_size_;
            int grid = (size + block - 1) / block;
            Kernels::GPU::sgd_step_kernel<<<grid, block>>>(d_weights_, d_gradients_,
                                                            learning_rate, size);
            cudaDeviceSynchronize();
            cudaMemcpy(weights, d_weights_, size * sizeof(float), cudaMemcpyDeviceToHost);
        }

#endif

        void SGD::update(float* weights, const float* gradients,
                         int size, float learning_rate)
        {
#ifdef USE_CUDA
            if (device_ == "gpu")
            {
                update_gpu(weights, gradients, size, learning_rate);
                return;
            }
#endif
            for (int i = 0; i < size; ++i)
                weights[i] -= learning_rate * gradients[i];
        }

    }
}
