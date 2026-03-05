/**
 * @file momentum.hpp
 * @brief SGD with Momentum optimizer
 */

#ifndef MOMENTUM_HPP
#define MOMENTUM_HPP

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
        class Momentum : public Optimizer
        {
        public:
            explicit Momentum(float mu = 0.9f, const std::string& device = "cpu");
            ~Momentum();

            void update(float* weights, const float* gradients,
                        int size, float learning_rate) override;

        private:
            float mu_;
            std::string device_;
            std::unordered_map<void*, std::vector<float>> vel_params_;

#ifdef USE_CUDA
            struct GpuState {
                float* d_vel = nullptr;
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

#endif // MOMENTUM_HPP
