/**
 * @file adam.hpp
 * @brief Adam optimizer
 */

#ifndef ADAM_HPP
#define ADAM_HPP

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "CppNet/optimizers/optimizer.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unordered_map>
#include <vector>
#include <string>

namespace CppNet
{
    namespace Optimizers
    {
        class Adam : public Optimizer
        {
        public:
            Adam(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f,
                 const std::string& device = "cpu");
            ~Adam();

            void update(float* weights, const float* gradients,
                        int size, float learning_rate) override;

        private:
            float beta1_;
            float beta2_;
            float epsilon_;
            std::string device_;
            int t_ = 0;
            std::unordered_map<void*, std::vector<float>> m_params_;
            std::unordered_map<void*, std::vector<float>> v_params_;
            std::unordered_map<void*, int> t_params_;

#ifdef USE_CUDA
            struct GpuState {
                float* d_m = nullptr;
                float* d_v = nullptr;
                int size = 0;
            };
            float* d_weights_ = nullptr;
            float* d_gradients_ = nullptr;
            int gpu_buf_ = 0;
            std::unordered_map<void*, GpuState> gpu_states_;

            void ensure_gpu(int size);
            void release_gpu();
            void update_gpu(float* weights, const float* gradients,
                            int size, float learning_rate);
#endif
        };
    }
}

#endif // ADAM_HPP
