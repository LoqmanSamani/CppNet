#ifndef LOSSES_HPP
#define LOSSES_HPP
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>



namespace CppNet
{
    namespace Losses
    {
        // base class for all losses
        class Loss
        {
            public:
                virtual ~Loss() = default;
                virtual double forward(const Eigen::Tensor<double, 2>& y, const Eigen::Tensor<double, 2>& y_hat) = 0;
                virtual Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& y_hat, const Eigen::Tensor<double, 2>& y) = 0;
        };

        class BinaryCrossEntropy
        {
            public:
                // compute binary cross-entropy loss between true labels y and predictions y_hat
                double forward(const Eigen::Tensor<double, 2>& y, const Eigen::Tensor<double, 2>& y_hat);
                // compute gradient of loss with respect to y_hat
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& y_hat, const Eigen::Tensor<double, 2>& y);
            private:
                // helper function to validate inputs
                void validate_inputs(const Eigen::Tensor<double, 2>& y, const Eigen::Tensor<double, 2>& y_hat);
                // helper function to clip values to avoid numerical issues
                //Eigen::Tensor<double, 2> clip_predictions(const Eigen::Tensor<double, 2>& y_hat, double eps = 1e-15);
        };

        class CategoricalCrossEntropy
        {
            public:
                // compute categorical cross-entropy loss between one-hot encoded labels y and softmax predictions y_hat
                double forward(const Eigen::Tensor<double, 2>& y, const Eigen::Tensor<double, 2>& y_hat);
                // compute gradient of loss with respect to y_hat
                Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2>& y_hat, const Eigen::Tensor<double, 2>& y);
            private:
                // helper function to validate inputs for categorical cross-entropy
                void validate_inputs(const Eigen::Tensor<double, 2>& y, const Eigen::Tensor<double, 2>& y_hat);
        };
    }
}
#endif // LOSSES_HPP

