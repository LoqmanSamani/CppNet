#ifndef CATEGORICAL_CROSS_ENTROPY_HPP
#define CATEGORICAL_CROSS_ENTROPY_HPP

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
        class CategoricalCrossEntropy : public Loss
        {
            
            public:

                CategoricalCrossEntropy(const std::string& reduction = "mean", bool from_logits = true,
                                        float label_smoothing = 0.0f, const std::string& device = "cpu");
                ~CategoricalCrossEntropy();
                
                // for classification: predictions are class probabilities/logits, targets are class indices or one-hot
                float forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets); // class indices
                float forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets); // one-hot
                float forward(const Eigen::Tensor<float, 3>& predictions, const Eigen::Tensor<int, 2>& targets); // sequence classification
                
                Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets);
                Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets);
                Eigen::Tensor<float, 3> backward(const Eigen::Tensor<float, 3>& predictions, const Eigen::Tensor<int, 2>& targets);

                static void set_num_threads(int num_threads);

            private:

                std::string reduction_;
                bool from_logits_;
                float label_smoothing_;
                std::string device_;
                Eigen::Tensor<float, 2> softmax_cache_;
                Eigen::Tensor<float, 2> targets_cache_;

                #ifdef USE_CUDA
                    float* d_pred_ = nullptr;
                    float* d_target_ = nullptr;
                    float* d_softmax_ = nullptr;
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

#endif // CATEGORICAL_CROSS_ENTROPY_HPP

