/**
 * @file softmax.cpp
 * @brief Softmax activation implementation
 *
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * Uses the max-subtract trick for numerical stability.
 */

#include "CppNet/activations/softmax.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Activations
    {
        Softmax::Softmax(const std::string& device)
            : device_(device)
        {
        }

        Eigen::Tensor<float, 2> Softmax::forward(const Eigen::Tensor<float, 2>& pre_activation)
        {
            int batch = pre_activation.dimension(0);
            int features = pre_activation.dimension(1);

            output_cache_2d_.resize(batch, features);

            #ifdef USE_OPENMP
            #pragma omp parallel for if(device_ == "cpu")
            #endif
            for (int i = 0; i < batch; ++i)
            {
                // Find max for numerical stability
                float max_val = pre_activation(i, 0);
                for (int j = 1; j < features; ++j)
                    max_val = std::max(max_val, pre_activation(i, j));

                // Compute exp(x - max) and sum
                float sum = 0.0f;
                for (int j = 0; j < features; ++j)
                {
                    output_cache_2d_(i, j) = std::exp(pre_activation(i, j) - max_val);
                    sum += output_cache_2d_(i, j);
                }

                // Normalize
                for (int j = 0; j < features; ++j)
                    output_cache_2d_(i, j) /= sum;
            }

            return output_cache_2d_;
        }

        Eigen::Tensor<float, 2> Softmax::backward(const Eigen::Tensor<float, 2>& grad_output)
        {
            // For softmax: dL/dx_i = sum_j(dL/dy_j * dy_j/dx_i)
            // dy_j/dx_i = y_i * (delta_ij - y_j)
            // Simplified: dL/dx = y * (dL/dy - sum_j(dL/dy_j * y_j))
            int batch = grad_output.dimension(0);
            int features = grad_output.dimension(1);

            Eigen::Tensor<float, 2> grad_input(batch, features);

            #ifdef USE_OPENMP
            #pragma omp parallel for if(device_ == "cpu")
            #endif
            for (int i = 0; i < batch; ++i)
            {
                float dot = 0.0f;
                for (int j = 0; j < features; ++j)
                    dot += grad_output(i, j) * output_cache_2d_(i, j);

                for (int j = 0; j < features; ++j)
                    grad_input(i, j) = output_cache_2d_(i, j) * (grad_output(i, j) - dot);
            }

            return grad_input;
        }

        Eigen::Tensor<float, 4> Softmax::forward(const Eigen::Tensor<float, 4>& pre_activation)
        {
            // Apply softmax along the last dimension (axis 3)
            int d0 = pre_activation.dimension(0);
            int d1 = pre_activation.dimension(1);
            int d2 = pre_activation.dimension(2);
            int d3 = pre_activation.dimension(3);

            output_cache_4d_.resize(d0, d1, d2, d3);

            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                    {
                        float max_val = pre_activation(a, b, c, 0);
                        for (int d = 1; d < d3; ++d)
                            max_val = std::max(max_val, pre_activation(a, b, c, d));

                        float sum = 0.0f;
                        for (int d = 0; d < d3; ++d)
                        {
                            output_cache_4d_(a, b, c, d) = std::exp(pre_activation(a, b, c, d) - max_val);
                            sum += output_cache_4d_(a, b, c, d);
                        }
                        for (int d = 0; d < d3; ++d)
                            output_cache_4d_(a, b, c, d) /= sum;
                    }

            return output_cache_4d_;
        }

        Eigen::Tensor<float, 4> Softmax::backward(const Eigen::Tensor<float, 4>& grad_output)
        {
            int d0 = grad_output.dimension(0);
            int d1 = grad_output.dimension(1);
            int d2 = grad_output.dimension(2);
            int d3 = grad_output.dimension(3);

            Eigen::Tensor<float, 4> grad_input(d0, d1, d2, d3);

            for (int a = 0; a < d0; ++a)
                for (int b = 0; b < d1; ++b)
                    for (int c = 0; c < d2; ++c)
                    {
                        float dot = 0.0f;
                        for (int d = 0; d < d3; ++d)
                            dot += grad_output(a, b, c, d) * output_cache_4d_(a, b, c, d);

                        for (int d = 0; d < d3; ++d)
                            grad_input(a, b, c, d) = output_cache_4d_(a, b, c, d) * (grad_output(a, b, c, d) - dot);
                    }

            return grad_input;
        }

        void Softmax::set_num_threads(int num_threads)
        {
            #ifdef USE_OPENMP
            omp_set_num_threads(num_threads);
            #endif
            (void)num_threads;
        }
    }
}
