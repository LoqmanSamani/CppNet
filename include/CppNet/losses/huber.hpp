/**
 * @file huber.hpp
 * @brief Huber (Smooth L1) loss
 */

#ifndef HUBER_HPP
#define HUBER_HPP

#include "CppNet/losses/loss.hpp"
#include <string>

namespace CppNet
{
    namespace Losses
    {
        /**
         * @class Huber
         * @brief Huber Loss: L = 0.5*(pred-target)^2 if |pred-target| < delta,
         *                       delta*(|pred-target| - 0.5*delta) otherwise.
         */
        class Huber : public Loss
        {
        public:
            explicit Huber(float delta = 1.0f, const std::string& reduction = "mean");

            float forward(const Eigen::Tensor<float, 2>& predictions,
                          const Eigen::Tensor<float, 2>& targets) override;

            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& predictions,
                                              const Eigen::Tensor<float, 2>& targets) override;

            static void set_num_threads(int num_threads);
            float get_delta() const { return delta_; }

        private:
            float delta_;
            std::string reduction_;
        };
    }
}

#endif // HUBER_HPP
