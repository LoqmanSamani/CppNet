/**
 * @file mae.cpp
 * @brief Mean Absolute Error loss implementation
 *
 * Forward: L = mean(|pred - target|)
 * Backward: dL/dpred = sign(pred - target) / N
 */

#include "CppNet/losses/mae.hpp"
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Losses
    {
        MAE::MAE(const std::string& reduction)
            : reduction_(reduction)
        {
        }

        float MAE::forward(const Eigen::Tensor<float, 2>& predictions,
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
                for (int j = 0; j < features; ++j)
                    sum += std::fabs(predictions(i, j) - targets(i, j));

            if (reduction_ == "mean")
                return sum / static_cast<float>(total);
            else if (reduction_ == "sum")
                return sum;
            else
                return sum;
        }

        Eigen::Tensor<float, 2> MAE::backward(const Eigen::Tensor<float, 2>& predictions,
                                               const Eigen::Tensor<float, 2>& targets)
        {
            int batch = predictions.dimension(0);
            int features = predictions.dimension(1);
            int total = batch * features;

            Eigen::Tensor<float, 2> grad(batch, features);

            float scale = (reduction_ == "mean") ? 1.0f / static_cast<float>(total) : 1.0f;

            #ifdef USE_OPENMP
            #pragma omp parallel for collapse(2)
            #endif
            for (int i = 0; i < batch; ++i)
                for (int j = 0; j < features; ++j)
                {
                    float diff = predictions(i, j) - targets(i, j);
                    if (diff > 0.0f)
                        grad(i, j) = scale;
                    else if (diff < 0.0f)
                        grad(i, j) = -scale;
                    else
                        grad(i, j) = 0.0f;
                }

            return grad;
        }

        void MAE::set_num_threads(int num_threads)
        {
            #ifdef USE_OPENMP
            omp_set_num_threads(num_threads);
            #endif
            (void)num_threads;
        }
    }
}
