/**
 * @file huber.cpp
 * @brief Huber (Smooth L1) loss implementation
 *
 * Forward: L = 0.5*(pred-target)^2         if |pred-target| <= delta
 *          L = delta*(|pred-target| - 0.5*delta) otherwise
 * Backward: dL/dpred = (pred-target)       if |pred-target| <= delta
 *           dL/dpred = delta*sign(pred-target) otherwise
 */

#include "CppNet/losses/huber.hpp"
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Losses
    {
        Huber::Huber(float delta, const std::string& reduction)
            : delta_(delta), reduction_(reduction)
        {
        }

        float Huber::forward(const Eigen::Tensor<float, 2>& predictions,
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
                    float abs_diff = std::fabs(diff);

                    if (abs_diff <= delta_)
                        sum += 0.5f * diff * diff;
                    else
                        sum += delta_ * (abs_diff - 0.5f * delta_);
                }
            }

            if (reduction_ == "mean")
                return sum / static_cast<float>(total);
            else if (reduction_ == "sum")
                return sum;
            else
                return sum;
        }

        Eigen::Tensor<float, 2> Huber::backward(const Eigen::Tensor<float, 2>& predictions,
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
                    float abs_diff = std::fabs(diff);

                    if (abs_diff <= delta_)
                        grad(i, j) = scale * diff;
                    else
                        grad(i, j) = scale * delta_ * ((diff > 0.0f) ? 1.0f : -1.0f);
                }

            return grad;
        }

        void Huber::set_num_threads(int num_threads)
        {
            #ifdef USE_OPENMP
            omp_set_num_threads(num_threads);
            #endif
            (void)num_threads;
        }
    }
}
