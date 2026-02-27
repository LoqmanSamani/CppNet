/**
 * @file schedulers.cpp
 * @brief Learning rate scheduler implementations
 */

#include "CppNet/utils/schedulers.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace CppNet
{
    namespace Schedulers
    {
        StepLR::StepLR(float initial_lr, int step_size, float gamma)
            : LRScheduler(initial_lr), step_size_(step_size), gamma_(gamma) {}

        float StepLR::step()
        {
            ++epoch_;
            current_lr_ = initial_lr_ * std::pow(gamma_, static_cast<float>(epoch_ / step_size_));
            return current_lr_;
        }

        ExponentialLR::ExponentialLR(float initial_lr, float gamma)
            : LRScheduler(initial_lr), gamma_(gamma) {}

        float ExponentialLR::step()
        {
            ++epoch_;
            current_lr_ = initial_lr_ * std::pow(gamma_, static_cast<float>(epoch_));
            return current_lr_;
        }

        CosineAnnealingLR::CosineAnnealingLR(float initial_lr, int T_max, float eta_min)
            : LRScheduler(initial_lr), T_max_(T_max), eta_min_(eta_min) {}

        float CosineAnnealingLR::step()
        {
            ++epoch_;
            current_lr_ = eta_min_ + 0.5f * (initial_lr_ - eta_min_) *
                           (1.0f + std::cos(static_cast<float>(M_PI) *
                                            static_cast<float>(epoch_) /
                                            static_cast<float>(T_max_)));
            return current_lr_;
        }
    }
}
