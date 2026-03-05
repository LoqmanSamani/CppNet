/**
 * @file adam.cpp
 * @brief Adam optimizer implementation
 *
 * m = beta1 * m + (1 - beta1) * dW
 * v = beta2 * v + (1 - beta2) * dW^2
 * m_hat = m / (1 - beta1^t)
 * v_hat = v / (1 - beta2^t)
 * W -= lr * m_hat / (sqrt(v_hat) + eps)
 */

#include "CppNet/optimizers/adam.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"
#include <cmath>

namespace CppNet
{
    namespace Optimizers
    {
        Adam::Adam(float beta1, float beta2, float epsilon, const std::string& device)
            : beta1_(beta1), beta2_(beta2), epsilon_(epsilon), device_(device), t_(0)
        {
        }

        Adam::~Adam()
        {
#ifdef USE_CUDA
            release_gpu();
#endif
        }

#ifdef USE_CUDA

        void Adam::ensure_gpu(int size)
        {
            if (gpu_buf_ >= size) return;
            release_gpu();
            cudaMalloc(&d_weights_, size * sizeof(float));
            cudaMalloc(&d_gradients_, size * sizeof(float));
            gpu_buf_ = size;
        }

        void Adam::release_gpu()
        {
            if (d_weights_)   { cudaFree(d_weights_);   d_weights_   = nullptr; }
            if (d_gradients_) { cudaFree(d_gradients_); d_gradients_ = nullptr; }
            gpu_buf_ = 0;
            for (auto& [key, state] : gpu_states_)
            {
                if (state.d_m) { cudaFree(state.d_m); state.d_m = nullptr; }
                if (state.d_v) { cudaFree(state.d_v); state.d_v = nullptr; }
            }
            gpu_states_.clear();
        }

        void Adam::update_gpu(float* weights, const float* gradients,
                               int size, float learning_rate)
        {
            ensure_gpu(size);
            void* key = static_cast<void*>(weights);

            if (gpu_states_.find(key) == gpu_states_.end())
            {
                GpuState state;
                state.size = size;
                cudaMalloc(&state.d_m, size * sizeof(float));
                cudaMalloc(&state.d_v, size * sizeof(float));
                cudaMemset(state.d_m, 0, size * sizeof(float));
                cudaMemset(state.d_v, 0, size * sizeof(float));
                gpu_states_[key] = state;
            }

            // Use per-key timestep from existing CPU map
            if (t_params_.find(key) == t_params_.end())
                t_params_[key] = 0;
            int& t = t_params_[key];
            ++t;

            float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t));
            float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t));

            auto& state = gpu_states_[key];

            cudaMemcpy(d_weights_, weights, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_gradients_, gradients, size * sizeof(float), cudaMemcpyHostToDevice);

            int block = 256;
            int grid = (size + block - 1) / block;
            Kernels::GPU::adam_step_kernel<<<grid, block>>>(
                d_weights_, d_gradients_, state.d_m, state.d_v,
                learning_rate, beta1_, beta2_, epsilon_, bc1, bc2, size);
            cudaDeviceSynchronize();
            cudaMemcpy(weights, d_weights_, size * sizeof(float), cudaMemcpyDeviceToHost);
        }

#endif

        void Adam::update(float* weights, const float* gradients,
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

            if (m_params_.find(key) == m_params_.end())
            {
                m_params_[key].assign(size, 0.0f);
                v_params_[key].assign(size, 0.0f);
                t_params_[key] = 0;
            }

            auto& m = m_params_[key];
            auto& v = v_params_[key];
            int& t = t_params_[key];
            ++t;

            float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t));
            float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t));

            for (int i = 0; i < size; ++i)
            {
                m[i] = beta1_ * m[i] + (1.0f - beta1_) * gradients[i];
                v[i] = beta2_ * v[i] + (1.0f - beta2_) * gradients[i] * gradients[i];
                float mh = m[i] / bc1;
                float vh = v[i] / bc2;
                weights[i] -= learning_rate * mh / (std::sqrt(vh) + epsilon_);
            }
        }
    }
}
