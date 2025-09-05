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
        class Loss
        {
        public:
            virtual ~Loss() = default;
        };

        class MSE : public Loss  // Mean Squared Error
        {
        private:
            std::string reduction; // "mean", "sum", or "none"
            
        public:
            MSE(const std::string& reduction = "mean");
            
            // Forward: compute loss value
            double forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            double forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            double forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets);
            
            // Backward: compute gradient with respect to predictions
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets);
        };

        class MAE : public Loss  // Mean Absolute Error
        {
        private:
            std::string reduction; // "mean", "sum", or "none"
            
        public:
            MAE(const std::string& reduction = "mean");
            
            double forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            double forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            double forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets);
            
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets);
        };

        class CrossEntropy : public Loss  // Cross Entropy Loss
        {
        private:
            std::string reduction; // "mean", "sum", or "none"
            bool from_logits; // Whether predictions are logits or probabilities
            double label_smoothing; // Label smoothing factor
            
        public:
            CrossEntropy(const std::string& reduction = "mean", bool from_logits = true, double label_smoothing = 0.0);
            
            // For classification: predictions are class probabilities/logits, targets are class indices or one-hot
            double forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets); // Class indices
            double forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets); // One-hot
            double forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<int, 2>& targets); // Sequence classification
            
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<int, 2>& targets);
        };

        class BinaryCrossEntropy : public Loss  // Binary Cross Entropy Loss
        {
        private:
            std::string reduction; // "mean", "sum", or "none"
            bool from_logits; // Whether predictions are logits or probabilities
            double pos_weight; // Weight for positive examples
            
        public:
            BinaryCrossEntropy(const std::string& reduction = "mean", bool from_logits = false, double pos_weight = 1.0);
            
            double forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            double forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
        };

        class HuberLoss : public Loss  // Huber Loss (smooth L1 loss)
        {
        private:
            double delta; // Threshold parameter
            std::string reduction; // "mean", "sum", or "none"
            
        public:
            HuberLoss(double delta = 1.0, const std::string& reduction = "mean");
            
            double forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            double forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            double forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets);
            
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets);
        };

        class KLDivergence : public Loss  // Kullback-Leibler Divergence Loss
        {
        private:
            std::string reduction; // "mean", "sum", or "none"
            bool log_target; // Whether target is in log space
            
        public:
            KLDivergence(const std::string& reduction = "mean", bool log_target = false);
            
            double forward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            double forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            double forward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets);
            
            Eigen::Tensor<double, 1> backward(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            Eigen::Tensor<double, 3> backward(const Eigen::Tensor<double, 3>& predictions, const Eigen::Tensor<double, 3>& targets);
        };

        class FocalLoss : public Loss  // Focal Loss for imbalanced classification
        {
        private:
            double alpha; // Weighting factor for rare class
            double gamma; // Focusing parameter
            std::string reduction; // "mean", "sum", or "none"
            
        public:
            FocalLoss(double alpha = 1.0, double gamma = 2.0, const std::string& reduction = "mean");
            
            double forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets);
            double forward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
            
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets);
            Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets);
        };
    }
}

#endif // LOSSES_HPP

