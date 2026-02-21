/**
 * @file loss.hpp
 * @brief Base class for all loss functions in CppNet
 */

#ifndef LOSS_HPP
#define LOSS_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace CppNet
{
    namespace Losses
    {
        /**
         * @class Loss
         * @brief Abstract base class for all loss functions.
         *
         * Subclasses must implement forward() and backward() methods.
         */
        class Loss
        {
        public:
            virtual ~Loss() = default;

            /**
             * @brief Compute the loss value
             * @param predictions Model output [batch, features]
             * @param targets Ground truth [batch, features]
             * @return Scalar loss value
             */
            virtual float forward(const Eigen::Tensor<float, 2>& predictions,
                                  const Eigen::Tensor<float, 2>& targets) = 0;

            /**
             * @brief Compute the gradient of the loss w.r.t. predictions
             * @param predictions Model output [batch, features]
             * @param targets Ground truth [batch, features]
             * @return Gradient tensor [batch, features]
             */
            virtual Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions,
                                                      const Eigen::Tensor<float, 2>& targets) = 0;
        };
    }
}

#endif // LOSS_HPP
