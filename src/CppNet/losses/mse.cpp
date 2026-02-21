/**
 * @file mse.cpp
 * @brief Mean Squared Error loss implementation
 *
 * Forward: L = mean((pred - target)^2)   [or sum/none]
 * Backward: dL/dpred = 2 * (pred - target) / N  [for mean reduction]
 */

#include "CppNet/losses/mse.hpp"
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Losses
    {
        MSE::MSE(const std::string& reduction)
            : reduction_(reduction)
        {
        }

        float MSE::forward(const Eigen::Tensor<float, 2>& predictions,
                           const Eigen::Tensor<float, 2>& targets)
        {
            int batch = predictions.dimension(0);
            int features = predictions.dimension(1);
            int total = batch * features;

            float sum = 0.0f;

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2) reduction(+:sum)
            #endif
            for (int i = 0; i < batch; ++i)
            {
                for (int j = 0; j < features; ++j)
                {
                    float diff = predictions(i, j) - targets(i, j);
                    sum += diff * diff;
                }
            }

            if (reduction_ == "mean")
                return sum / static_cast<float>(total);
            else if (reduction_ == "sum")
                return sum;
            else
                return sum; // "none" returns total (individual losses would require tensor return)
        }

        Eigen::Tensor<float, 2> MSE::backward(const Eigen::Tensor<float, 2>& predictions,
                                               const Eigen::Tensor<float, 2>& targets)
        {
            int batch = predictions.dimension(0);
            int features = predictions.dimension(1);
            int total = batch * features;

            Eigen::Tensor<float, 2> grad(batch, features);

            float scale = (reduction_ == "mean") ? 2.0f / static_cast<float>(total) : 2.0f;

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2)
            #endif
            for (int i = 0; i < batch; ++i)
                for (int j = 0; j < features; ++j)
                    grad(i, j) = scale * (predictions(i, j) - targets(i, j));

            return grad;
        }

        void MSE::set_num_threads(int num_threads)
        {
            #ifdef USE_OPENMP
            omp_set_num_threads(num_threads);
            #endif
            (void)num_threads;
        }
    }
}
