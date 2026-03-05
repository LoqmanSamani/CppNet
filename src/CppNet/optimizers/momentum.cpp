/**
 * @file momentum.cpp
 * @brief SGD with Momentum optimizer implementation
 *
 * v = mu * v - lr * dW
 * W += v
 */

#include "CppNet/optimizers/momentum.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"

namespace CppNet
{
    namespace Optimizers
    {
        Momentum::Momentum(float mu, const std::string& device)
            : mu_(mu), device_(device)
        {
        }

        Momentum::~Momentum()
        {
#ifdef USE_CUDA
            release_gpu();
#endif
        }

#ifdef USE_CUDA

        void Momentum::ensure_gpu(int size)
        {
            if (gpu_buf_ >= size) return;
            release_gpu();
            cudaMalloc(&d_weights_, size * sizeof(float));
            cudaMalloc(&d_gradients_, size * sizeof(float));
            gpu_buf_ = size;
        }

        void Momentum::release_gpu()
        {
            if (d_weights_)   { cudaFree(d_weights_);   d_weights_   = nullptr; }
            if (d_gradients_) { cudaFree(d_gradients_); d_gradients_ = nullptr; }
            gpu_buf_ = 0;
            for (auto& [key, state] : gpu_states_)
            {
                if (state.d_vel) { cudaFree(state.d_vel); state.d_vel = nullptr; }
            }
            gpu_states_.clear();
        }

        void Momentum::update_gpu(float* weights, const float* gradients,
                                   int size, float learning_rate)
        {
            ensure_gpu(size);
            void* key = static_cast<void*>(weights);

            if (gpu_states_.find(key) == gpu_states_.end())
            {
                GpuState state;
                state.size = size;
                cudaMalloc(&state.d_vel, size * sizeof(float));
                cudaMemset(state.d_vel, 0, size * sizeof(float));
                gpu_states_[key] = state;
            }

            auto& state = gpu_states_[key];

            cudaMemcpy(d_weights_, weights, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_gradients_, gradients, size * sizeof(float), cudaMemcpyHostToDevice);

            int block = 256;
            int grid = (size + block - 1) / block;
            Kernels::GPU::momentum_step_kernel<<<grid, block>>>(
                d_weights_, d_gradients_, state.d_vel, mu_, learning_rate, size);
            cudaDeviceSynchronize();
            cudaMemcpy(weights, d_weights_, size * sizeof(float), cudaMemcpyDeviceToHost);
        }

#endif

        void Momentum::update(float* weights, const float* gradients,
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

            if (vel_params_.find(key) == vel_params_.end())
                vel_params_[key].assign(size, 0.0f);

            auto& vel = vel_params_[key];

            for (int i = 0; i < size; ++i)
            {
                vel[i] = mu_ * vel[i] - learning_rate * gradients[i];
                weights[i] += vel[i];
            }
        }
    }
}
