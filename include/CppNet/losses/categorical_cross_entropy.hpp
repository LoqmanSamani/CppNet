#ifndef LOSSES_HPP
#define LOSSES_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <vector>




namespace CppNet
{
    namespace Losses
    {
        // base class for all losses
        class Loss
        {
            public:
                virtual ~Loss() = default;
                virtual float forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets) = 0;
                virtual Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets) = 0;
        };
       
        class CategoricalCrossEntropy : public Loss
        {
            
            public:

                CategoricalCrossEntropy(const std::string& reduction = "mean", bool from_logits = true, float label_smoothing = 0.0f);
                
                // for classification: predictions are class probabilities/logits, targets are class indices or one-hot
                float forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets); // class indices
                float forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets); // one-hot
                float forward(const Eigen::Tensor<float, 3>& predictions, const Eigen::Tensor<int, 2>& targets); // sequence classification
                
                Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets);
                Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets);
                Eigen::Tensor<float, 3> backward(const Eigen::Tensor<float, 3>& predictions, const Eigen::Tensor<int, 2>& targets);

                static void set_num_threads(int num_threads);

            private:

                std::string reduction_; // "mean", "sum", or "none"
                bool from_logits_; // whether predictions are logits or probabilities
                float label_smoothing_; // label smoothing factor
                Eigen::Tensor<float, 2> softmax_cache_;  // cache softmax output
                Eigen::Tensor<float, 2> targets_cache_;  // cache processed targets
        };
    }
}

#endif

