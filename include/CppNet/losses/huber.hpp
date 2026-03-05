/**
 * @file huber.hpp
 * @brief Huber (Smooth L1) loss
 */

#ifndef HUBER_HPP
#define HUBER_HPP

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
         * @class Huber
         * @brief Huber Loss: L = 0.5*(pred-target)^2 if |pred-target| < delta,
         *                       delta*(|pred-target| - 0.5*delta) otherwise.
         */
        class Huber : public Loss
        {
        public:
            explicit Huber(float delta = 1.0f, const std::string& reduction = "mean",
                           const std::string& device = "cpu");
            ~Huber();

            float forward(const Eigen::Tensor<float, 2>& predictions,
                          const Eigen::Tensor<float, 2>& targets) override;

            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions,
                                              const Eigen::Tensor<float, 2>& targets) override;

            static void set_num_threads(int num_threads);
            float get_delta() const { return delta_; }

        private:
            float delta_;
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

#endif // HUBER_HPP