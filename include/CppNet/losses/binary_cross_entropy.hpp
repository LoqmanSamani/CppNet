#ifndef BINARY_CROSS_ENTROPY_HPP
#define BINARY_CROSS_ENTROPY_HPP

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
        class BinaryCrossEntropy : public Loss
        {
            public:

                BinaryCrossEntropy(const std::string& reduction = "mean", bool from_logits = false, float pos_weight = 1.0f);

                float forward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets);
                Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets);

                static void set_num_threads(int num_threads);

            private:
                std::string reduction_; // "mean", "sum", or "none"
                bool from_logits_; // whether predictions are logits or probabilities
                float pos_weight_; // weight for positive examples
                // helper function to validate inputs
                void validate_inputs(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets);   
        };
    }
}

#endif // BINARY_CROSS_ENTROPY_HPP