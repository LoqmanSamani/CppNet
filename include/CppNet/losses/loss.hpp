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

            virtual float forward(const Eigen::Tensor<float, 2>& predictions,
                                  const Eigen::Tensor<float, 2>& targets) = 0;

            virtual Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions,
                                                      const Eigen::Tensor<float, 2>& targets) = 0;
        };
    }
}

#endif // LOSS_HPP
