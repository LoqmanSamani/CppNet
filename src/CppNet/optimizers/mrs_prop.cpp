/**
 * @file mrs_prop.cpp
 * @brief RMSProp optimizer implementation
 *
 * s = rho * s + (1 - rho) * dW^2
 * W -= lr * dW / (sqrt(s) + eps)
 */

#include "CppNet/optimizers/mrs_prop.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"
#include <cmath>

namespace CppNet
{
    namespace Optimizers
    {
        RMSProp::RMSProp(float rho, float epsilon, const std::string& device)
            : rho_(rho), epsilon_(epsilon), device_(device)
        {
        }

        RMSProp::~RMSProp()
        {
#ifdef USE_CUDA
            release_gpu();
#endif
        }

#ifdef USE_CUDA

        void RMSProp::ensure_gpu(int size)
        {
            if (gpu_buf_ >= size) return;
            release_gpu();
            cudaMalloc(&d_weights_, size * sizeof(float));
            cudaMalloc(&d_gradients_, size * sizeof(float));
            gpu_buf_ = size;
        }

        void RMSProp::release_gpu()
        {
            if (d_weights_)   { cudaFree(d_weights_);   d_weights_   = nullptr; }
            if (d_gradients_) { cudaFree(d_gradients_); d_gradients_ = nullptr; }
            gpu_buf_ = 0;
            for (auto& [key, state] : gpu_states_)
            {
                if (state.d_cache) { cudaFree(state.d_cache); state.d_cache = nullptr; }
            }
            gpu_states_.clear();
        }

        void RMSProp::update_gpu(float* weights, const float* gradients,
                                  int size, float learning_rate)
        {
            ensure_gpu(size);
            void* key = static_cast<void*>(weights);

            if (gpu_states_.find(key) == gpu_states_.end())
            {
                GpuState state;
                state.size = size;
                cudaMalloc(&state.d_cache, size * sizeof(float));
                cudaMemset(state.d_cache, 0, size * sizeof(float));
                gpu_states_[key] = state;
            }

            auto& state = gpu_states_[key];

            cudaMemcpy(d_weights_, weights, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_gradients_, gradients, size * sizeof(float), cudaMemcpyHostToDevice);

            int block = 256;
            int grid = (size + block - 1) / block;
            Kernels::GPU::rmsprop_step_kernel<<<grid, block>>>(
                d_weights_, d_gradients_, state.d_cache,
                learning_rate, rho_, epsilon_, size);
            cudaDeviceSynchronize();
            cudaMemcpy(weights, d_weights_, size * sizeof(float), cudaMemcpyDeviceToHost);
        }

#endif

        void RMSProp::update(float* weights, const float* gradients,
                             int size, float learning_rate)
        {
#ifdef USE_CUDA
            if (device_ == "gpu")
            {
                update_gpu(weights, gradients, size, learning_rate);
                return;
            }
#endif
            void* key = static_cast<void*>(weights);

            if (cache_params_.find(key) == cache_params_.end())
                cache_params_[key].assign(size, 0.0f);

            auto& s = cache_params_[key];

            for (int i = 0; i < size; ++i)
            {
                s[i] = rho_ * s[i] + (1.0f - rho_) * gradients[i] * gradients[i];
                weights[i] -= learning_rate * gradients[i] / (std::sqrt(s[i]) + epsilon_);
            }
        }
    }
}
