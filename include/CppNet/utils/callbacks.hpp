/**
 * @file callbacks.hpp
 * @brief Training callbacks for CppNet
 *
 * - EarlyStopping: Monitor a metric (e.g. validation loss) and stop
 *   training when it has stopped improving for `patience` epochs.
 */

#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP

#include <string>
#include <limits>
#include <cmath>
#include <iostream>

namespace CppNet
{
    namespace Callbacks
    {
        /**
         * @class EarlyStopping
         * @brief Stop training when a monitored metric has stopped improving
         *
         * @example
         *   EarlyStopping es(/* patience */ 5, /* min_delta */ 1e-4f);
         *   for (int epoch = 0; epoch < max_epochs; ++epoch) {
         *       float val_loss = train_one_epoch(...);
         *       if (es.step(val_loss)) {
         *           std::cout << "Early stopping at epoch " << epoch << "\n";
         *           break;
         *       }
         *   }
         */
        class EarlyStopping
        {
        public:
            /**
             * @param patience   Number of epochs with no improvement before stopping
             * @param min_delta  Minimum change to qualify as an improvement (default 0)
             * @param mode       "min" (lower is better, e.g. loss) or
             *                   "max" (higher is better, e.g. accuracy)
             * @param verbose    If true, print messages when patience counter changes
             */
            EarlyStopping(int patience = 5, float min_delta = 0.0f,
                          const std::string& mode = "min", bool verbose = false);

            /**
             * @brief Report the latest metric value
             * @param metric Current epoch's metric value
             * @return true if training should stop, false otherwise
             */
            bool step(float metric);

            /** Reset internal state so the callback can be reused */
            void reset();

            int   get_patience() const { return patience_; }
            int   get_wait() const { return wait_; }
            float get_best() const { return best_; }
            int   get_best_epoch() const { return best_epoch_; }
            bool  should_stop() const { return stopped_; }

        private:
            int   patience_;
            float min_delta_;
            std::string mode_;
            bool  verbose_;

            float best_;
            int   wait_;
            int   epoch_;
            int   best_epoch_;
            bool  stopped_;

            bool is_improvement(float metric) const;
        };
    }
}

#endif // CALLBACKS_HPP
