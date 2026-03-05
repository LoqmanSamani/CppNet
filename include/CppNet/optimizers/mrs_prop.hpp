/**
 * @file mrs_prop.hpp
 * @brief RMSProp optimizer
 */

#ifndef RMS_PROP_HPP
#define RMS_PROP_HPP

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
        class RMSProp : public Optimizer
        {
        public:
            RMSProp(float rho = 0.9f, float epsilon = 1e-8f,
                    const std::string& device = "cpu");
            ~RMSProp();

            void update(float* weights, const float* gradients,
                        int size, float learning_rate) override;

        private:
            float rho_;
            float epsilon_;
            std::string device_;
            std::unordered_map<void*, std::vector<float>> cache_params_;

#ifdef USE_CUDA
            struct GpuState {
                float* d_cache = nullptr;
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

#endif // RMS_PROP_HPP
