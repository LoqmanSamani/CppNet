/**
 * @file schedulers.hpp
 * @brief Learning rate schedulers for CppNet
 *
 * - StepLR:          Decay LR by gamma every step_size epochs
 * - ExponentialLR:   Decay LR by gamma every epoch  (lr * gamma^epoch)
 * - CosineAnnealingLR: Cosine annealing from initial LR to eta_min
 */

#ifndef SCHEDULERS_HPP
#define SCHEDULERS_HPP

#include <cmath>
#include <algorithm>
#include <string>

namespace CppNet
{
    namespace Schedulers
    {
        /**
         * @class LRScheduler
         * @brief Abstract base class for learning-rate schedulers
         */
        class LRScheduler
        {
        public:
            explicit LRScheduler(float initial_lr)
                : initial_lr_(initial_lr), current_lr_(initial_lr), epoch_(0) {}
            virtual ~LRScheduler() = default;

            /// Advance one epoch and return the new LR
            virtual float step() = 0;

            float get_lr() const { return current_lr_; }
            float get_initial_lr() const { return initial_lr_; }
            int   get_epoch() const { return epoch_; }

        protected:
            float initial_lr_;
            float current_lr_;
            int   epoch_;
        };

        // -----------------------------------------------------------------
        //  StepLR
        // -----------------------------------------------------------------

        /**
         * @class StepLR
         * @brief Decay learning rate by `gamma` every `step_size` epochs
         *
         * lr = initial_lr * gamma^(epoch / step_size)
         */
        class StepLR : public LRScheduler
        {
        public:
            /**
             * @param initial_lr  Starting learning rate
             * @param step_size   Period (in epochs) of each decay
             * @param gamma       Multiplicative decay factor (default 0.1)
             */
            StepLR(float initial_lr, int step_size, float gamma = 0.1f);
            ~StepLR() override = default;

            float step() override;

            int   get_step_size() const { return step_size_; }
            float get_gamma() const { return gamma_; }

        private:
            int   step_size_;
            float gamma_;
        };

        // -----------------------------------------------------------------
        //  ExponentialLR
        // -----------------------------------------------------------------

        /**
         * @class ExponentialLR
         * @brief Decay learning rate exponentially every epoch
         *
         * lr = initial_lr * gamma^epoch
         */
        class ExponentialLR : public LRScheduler
        {
        public:
            /**
             * @param initial_lr Starting learning rate
             * @param gamma      Multiplicative decay per epoch (e.g. 0.95)
             */
            ExponentialLR(float initial_lr, float gamma);
            ~ExponentialLR() override = default;

            float step() override;

            float get_gamma() const { return gamma_; }

        private:
            float gamma_;
        };

        // -----------------------------------------------------------------
        //  CosineAnnealingLR
        // -----------------------------------------------------------------

        /**
         * @class CosineAnnealingLR
         * @brief Cosine annealing schedule
         *
         * lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max))
         */
        class CosineAnnealingLR : public LRScheduler
        {
        public:
            /**
             * @param initial_lr Starting (maximum) learning rate
             * @param T_max      Maximum number of epochs (half period)
             * @param eta_min    Minimum learning rate (default 0)
             */
            CosineAnnealingLR(float initial_lr, int T_max, float eta_min = 0.0f);
            ~CosineAnnealingLR() override = default;

            float step() override;

            int   get_T_max() const { return T_max_; }
            float get_eta_min() const { return eta_min_; }

        private:
            int   T_max_;
            float eta_min_;
        };
    }
}

#endif // SCHEDULERS_HPP
