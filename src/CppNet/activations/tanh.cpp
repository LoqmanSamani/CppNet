/**
 * @file tanh.cpp
 * @brief Tanh activation implementation
 *
 * tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * Backward: dL/dx = dL/dy * (1 - tanh(x)^2)
 */

#include "CppNet/activations/tanh.hpp"
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Activations
    {
        Tanh::Tanh(const std::string& device)
            : device_(device)
        {
        }

        Eigen::Tensor<float, 2> Tanh::forward(const Eigen::Tensor<float, 2>& pre_activation)
        {
            int rows = pre_activation.dimension(0);
            int cols = pre_activation.dimension(1);

            output_cache_2d_.resize(rows, cols);

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2) if(device_ == "cpu")
            #endif
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    output_cache_2d_(i, j) = std::tanh(pre_activation(i, j));

            return output_cache_2d_;
        }

        Eigen::Tensor<float, 2> Tanh::backward(const Eigen::Tensor<float, 2>& grad_output)
        {
            int rows = grad_output.dimension(0);
            int cols = grad_output.dimension(1);

            Eigen::Tensor<float, 2> grad_input(rows, cols);

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2) if(device_ == "cpu")
            #endif
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                {
                    float t = output_cache_2d_(i, j);
                    grad_input(i, j) = grad_output(i, j) * (1.0f - t * t);
                }

            return grad_input;
        }

        Eigen::Tensor<float, 4> Tanh::forward(const Eigen::Tensor<float, 4>& pre_activation)
        {
            int d0 = pre_activation.dimension(0);
            int d1 = pre_activation.dimension(1);
            int d2 = pre_activation.dimension(2);
            int d3 = pre_activation.dimension(3);

            output_cache_4d_.resize(d0, d1, d2, d3);

            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                        for (int d = 0; d < d3; ++d)
                            output_cache_4d_(a, b, c, d) = std::tanh(pre_activation(a, b, c, d));

            return output_cache_4d_;
        }

        Eigen::Tensor<float, 4> Tanh::backward(const Eigen::Tensor<float, 4>& grad_output)
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
                        {
                            float t = output_cache_4d_(a, b, c, d);
                            grad_input(a, b, c, d) = grad_output(a, b, c, d) * (1.0f - t * t);
                        }

            return grad_input;
        }

        void Tanh::set_num_threads(int num_threads)
        {
            #ifdef USE_OPENMP
            omp_set_num_threads(num_threads);
            #endif
            (void)num_threads;
        }
    }
}
