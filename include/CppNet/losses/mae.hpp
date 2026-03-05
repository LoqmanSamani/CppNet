/**
 * @file mae.hpp
 * @brief Mean Absolute Error loss
 */

#ifndef MAE_HPP
#define MAE_HPP

#include "CppNet/losses/loss.hpp"
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace CppNet
{
    namespace Losses
    {
        /**
         * @class MAE
         * @brief Mean Absolute Error: L = mean(|pred - target|)
         */
        class MAE : public Loss
        {
        public:
            explicit MAE(const std::string& reduction = "mean",
                         const std::string& device = "cpu");
            ~MAE();

            float forward(const Eigen::Tensor<float, 2>& predictions,
                          const Eigen::Tensor<float, 2>& targets) override;

            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions,
                                              const Eigen::Tensor<float, 2>& targets) override;

            static void set_num_threads(int num_threads);

        private:
            std::string reduction_;
            std::string device_;

            #ifdef USE_CUDA
                float* d_pred_ = nullptr;
                float* d_target_ = nullptr;
                float* d_loss_ = nullptr;
                float* d_grad_ = nullptr;
                std::size_t gpu_buf_ = 0;
                bool gpu_init_ = false;
                void ensure_gpu(std::size_t n);
                void release_gpu();
                float forward_gpu(const Eigen::Tensor<float, 2>& predictions,
                                  const Eigen::Tensor<float, 2>& targets);
                void backward_gpu(const Eigen::Tensor<float, 2>& predictions,
                                  const Eigen::Tensor<float, 2>& targets,
                                  Eigen::Tensor<float, 2>& grad);
            #endif
        };
    }
}

#endif // MAE_HPP
