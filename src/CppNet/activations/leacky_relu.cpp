/**
 * @file leacky_relu.cpp
 * @brief Leaky ReLU activation implementation
 *
 * f(x) = x if x > 0, else alpha * x
 * Backward: dL/dx = dL/dy if x > 0, else alpha * dL/dy
 */

#include "CppNet/activations/leacky_relu.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Activations
    {
        LeakyReLU::LeakyReLU(float alpha, const std::string& device)
            : alpha_(alpha), device_(device)
        {
        }

        Eigen::Tensor<float, 2> LeakyReLU::forward(const Eigen::Tensor<float, 2>& pre_activation)
        {
            int rows = pre_activation.dimension(0);
            int cols = pre_activation.dimension(1);

            input_cache_2d_ = pre_activation;
            Eigen::Tensor<float, 2> output(rows, cols);

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2) if(device_ == "cpu")
            #endif
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    output(i, j) = (pre_activation(i, j) > 0.0f)
                        ? pre_activation(i, j)
                        : alpha_ * pre_activation(i, j);

            return output;
        }

        Eigen::Tensor<float, 2> LeakyReLU::backward(const Eigen::Tensor<float, 2>& grad_output)
        {
            int rows = grad_output.dimension(0);
            int cols = grad_output.dimension(1);

            Eigen::Tensor<float, 2> grad_input(rows, cols);

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2) if(device_ == "cpu")
            #endif
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    grad_input(i, j) = (input_cache_2d_(i, j) > 0.0f)
                        ? grad_output(i, j)
                        : alpha_ * grad_output(i, j);

            return grad_input;
        }

        Eigen::Tensor<float, 4> LeakyReLU::forward(const Eigen::Tensor<float, 4>& pre_activation)
        {
            int d0 = pre_activation.dimension(0);
            int d1 = pre_activation.dimension(1);
            int d2 = pre_activation.dimension(2);
            int d3 = pre_activation.dimension(3);

            input_cache_4d_ = pre_activation;
            Eigen::Tensor<float, 4> output(d0, d1, d2, d3);

            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                        for (int d = 0; d < d3; ++d)
                            output(a, b, c, d) = (pre_activation(a, b, c, d) > 0.0f)
                                ? pre_activation(a, b, c, d)
                                : alpha_ * pre_activation(a, b, c, d);

            return output;
        }

        Eigen::Tensor<float, 4> LeakyReLU::backward(const Eigen::Tensor<float, 4>& grad_output)
        {
            int d0 = grad_output.dimension(0);
            int d1 = grad_output.dimension(1);
            int d2 = grad_output.dimension(2);
            int d3 = grad_output.dimension(3);

            Eigen::Tensor<float, 4> grad_input(d0, d1, d2, d3);

            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                        for (int d = 0; d < d3; ++d)
                            grad_input(a, b, c, d) = (input_cache_4d_(a, b, c, d) > 0.0f)
                                ? grad_output(a, b, c, d)
                                : alpha_ * grad_output(a, b, c, d);

            return grad_input;
        }

        void LeakyReLU::set_num_threads(int num_threads)
        {
            #ifdef USE_OPENMP
            omp_set_num_threads(num_threads);
            #endif
            (void)num_threads;
        }
    }
}
