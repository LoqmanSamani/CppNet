/**
 * @file mse.hpp
 * @brief Mean Squared Error loss
 */

#ifndef MSE_HPP
#define MSE_HPP

#include "CppNet/losses/loss.hpp"
#include <string>

namespace CppNet
{
    namespace Losses
    {
        /**
         * @class MSE
         * @brief Mean Squared Error: L = mean((pred - target)^2)
         */
        class MSE : public Loss
        {
        public:
            explicit MSE(const std::string& reduction = "mean");

            float forward(const Eigen::Tensor<float, 2>& predictions,
                          const Eigen::Tensor<float, 2>& targets) override;

            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions,
                                              const Eigen::Tensor<float, 2>& targets) override;

            static void set_num_threads(int num_threads);

        private:
            std::string reduction_;
        };
    }
}

#endif // MSE_HPP
