#ifndef BINARY_CROSS_ENTROPY_HPP
#define BINARY_CROSS_ENTROPY_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <vector>
#include "CppNet/losses/loss.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace CppNet
{
    namespace Losses
    {
        class BinaryCrossEntropy : public Loss
        {
            public:

                BinaryCrossEntropy(const std::string& reduction = "mean", bool from_logits = false,
                                   float pos_weight = 1.0f, const std::string& device = "cpu");
                ~BinaryCrossEntropy();

                float forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets);
                Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets);

                static void set_num_threads(int num_threads);

            private:
                std::string reduction_;
                bool from_logits_;
                float pos_weight_;
                std::string device_;
                void validate_inputs(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets);

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

#endif // BINARY_CROSS_ENTROPY_HPP