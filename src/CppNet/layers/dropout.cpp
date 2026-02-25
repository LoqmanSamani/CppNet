/**
 * @file dropout.cpp
 * @brief Dropout layer implementation (inverted dropout)
 */

#include "CppNet/layers/dropout.hpp"
#include <chrono>
#include <stdexcept>

namespace CppNet
{
    namespace Layers
    {
        Dropout::Dropout(float p)
            : p_(p),
              scale_(1.0f / (1.0f - p)),
              gen_(static_cast<unsigned>(
                  std::chrono::steady_clock::now().time_since_epoch().count())),
              dist_(1.0 - static_cast<double>(p))
        {
            if (p_ < 0.0f || p_ >= 1.0f)
                throw std::invalid_argument("Dropout probability must be in [0, 1)");
        }

        Dropout::~Dropout() = default;

        // ---------- 2D forward / backward ----------

        Eigen::Tensor<float, 2> Dropout::forward(
            const Eigen::Tensor<float, 2>& input)
        {
            if (!training_ || p_ == 0.0f)
                return input;

            int rows = input.dimension(0);
            int cols = input.dimension(1);

            mask_2d_.resize(rows, cols);
            Eigen::Tensor<float, 2> output(rows, cols);

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                {
                    float m = dist_(gen_) ? 1.0f : 0.0f;
                    mask_2d_(i, j) = m;
                    output(i, j) = input(i, j) * m * scale_;
                }

            return output;
        }

        Eigen::Tensor<float, 2> Dropout::backward(
            const Eigen::Tensor<float, 2>& grad_output)
        {
            if (!training_ || p_ == 0.0f)
                return grad_output;

            int rows = grad_output.dimension(0);
            int cols = grad_output.dimension(1);
            Eigen::Tensor<float, 2> grad_input(rows, cols);

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    grad_input(i, j) = grad_output(i, j) * mask_2d_(i, j) * scale_;

            return grad_input;
        }

        // ---------- 4D forward / backward ----------

        Eigen::Tensor<float, 4> Dropout::forward(
            const Eigen::Tensor<float, 4>& input)
        {
            if (!training_ || p_ == 0.0f)
                return input;

            int d0 = input.dimension(0);
            int d1 = input.dimension(1);
            int d2 = input.dimension(2);
            int d3 = input.dimension(3);

            mask_4d_.resize(d0, d1, d2, d3);
            Eigen::Tensor<float, 4> output(d0, d1, d2, d3);

            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                        for (int d = 0; d < d3; ++d)
                        {
                            float m = dist_(gen_) ? 1.0f : 0.0f;
                            mask_4d_(a, b, c, d) = m;
                            output(a, b, c, d) = input(a, b, c, d) * m * scale_;
                        }

            return output;
        }

        Eigen::Tensor<float, 4> Dropout::backward(
            const Eigen::Tensor<float, 4>& grad_output)
        {
            if (!training_ || p_ == 0.0f)
                return grad_output;

            int d0 = grad_output.dimension(0);
            int d1 = grad_output.dimension(1);
            int d2 = grad_output.dimension(2);
            int d3 = grad_output.dimension(3);

            Eigen::Tensor<float, 4> grad_input(d0, d1, d2, d3);

            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                        for (int d = 0; d < d3; ++d)
                            grad_input(a, b, c, d) =
                                grad_output(a, b, c, d) * mask_4d_(a, b, c, d) * scale_;

            return grad_input;
        }

        void Dropout::step(Optimizers::Optimizer& /*optimizer*/, float /*learning_rate*/)
        {
            // No parameters to update
        }
    }
}
