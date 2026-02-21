/**
 * @file callbacks.cpp
 * @brief EarlyStopping callback implementation
 */

#include "CppNet/utils/callbacks.hpp"

namespace CppNet
{
    namespace Callbacks
    {
        EarlyStopping::EarlyStopping(int patience, float min_delta,
                                     const std::string& mode, bool verbose)
            : patience_(patience), min_delta_(std::abs(min_delta)),
              mode_(mode), verbose_(verbose),
              wait_(0), epoch_(0), best_epoch_(0), stopped_(false)
        {
            if (mode_ == "min")
                best_ = std::numeric_limits<float>::infinity();
            else
                best_ = -std::numeric_limits<float>::infinity();
        }

        bool EarlyStopping::step(float metric)
        {
            ++epoch_;

            if (is_improvement(metric))
            {
                best_ = metric;
                best_epoch_ = epoch_;
                wait_ = 0;

                if (verbose_)
                    std::cout << "[EarlyStopping] Epoch " << epoch_
                              << ": metric improved to " << best_ << "\n";
            }
            else
            {
                ++wait_;

                if (verbose_)
                    std::cout << "[EarlyStopping] Epoch " << epoch_
                              << ": no improvement (" << wait_ << "/" << patience_ << ")\n";

                if (wait_ >= patience_)
                {
                    stopped_ = true;
                    if (verbose_)
                        std::cout << "[EarlyStopping] Stopping. Best was "
                                  << best_ << " at epoch " << best_epoch_ << "\n";
                    return true;
                }
            }

            return false;
        }

        void EarlyStopping::reset()
        {
            wait_ = 0;
            epoch_ = 0;
            best_epoch_ = 0;
            stopped_ = false;

            if (mode_ == "min")
                best_ = std::numeric_limits<float>::infinity();
            else
                best_ = -std::numeric_limits<float>::infinity();
        }

        bool EarlyStopping::is_improvement(float metric) const
        {
            if (mode_ == "min")
                return metric < (best_ - min_delta_);
            else
                return metric > (best_ + min_delta_);
        }
    }
}
