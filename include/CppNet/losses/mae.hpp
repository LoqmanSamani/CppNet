/**
 * @file mae.hpp
 * @brief Mean Absolute Error loss
 */

#ifndef MAE_HPP
#define MAE_HPP

#include "CppNet/losses/loss.hpp"
#include <string>

namespace CppNet
{
    namespace Losses
    {
        /**
         * @class MAE
         * @brief Mean Absolute Error: L = mean(|pred - target|)
         */
        class MAE : public Loss
        {
        public:
            explicit MAE(const std::string& reduction = "mean");

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

#endif // MAE_HPP
