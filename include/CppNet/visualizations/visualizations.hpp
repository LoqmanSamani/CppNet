/**
 * @file visualizations.hpp
 * @brief Training history logging and export for CppNet
 *
 * Provides a TrainingLogger that records per-epoch loss and metric
 * values and can export them to CSV for external plotting.
 */

#ifndef VISUALIZATIONS_HPP
#define VISUALIZATIONS_HPP

#include <vector>
#include <string>
#include <map>
#include <fstream>

namespace CppNet
{
    namespace Visualizations
    {
        /**
         * @class TrainingLogger
         * @brief Records training and validation metrics per epoch
         *
         * @example
         *   TrainingLogger logger;
         *   for (int epoch = 0; epoch < 100; ++epoch) {
         *       float loss = train_one_epoch();
         *       logger.log("train_loss", loss);
         *       logger.next_epoch();
         *   }
         *   logger.export_csv("training_history.csv");
         */
        class TrainingLogger
        {
        public:
            TrainingLogger() = default;

            /**
             * @brief Log a metric value for the current epoch
             * @param name Metric name (e.g., "train_loss", "val_accuracy")
             * @param value Metric value
             */
            void log(const std::string& name, float value);

            /**
             * @brief Advance to the next epoch
             */
            void next_epoch();

            /**
             * @brief Get the current epoch number (0-based)
             * @return Current epoch
             */
            int current_epoch() const { return current_epoch_; }

            /**
             * @brief Get all logged values for a given metric
             * @param name Metric name
             * @return Vector of (epoch, value) pairs
             */
            std::vector<std::pair<int, float>> get_history(const std::string& name) const;

            /**
             * @brief Get all metric names that have been logged
             * @return Vector of metric name strings
             */
            std::vector<std::string> get_metric_names() const;

            /**
             * @brief Export all logged metrics to a CSV file
             * @param filepath Output CSV file path
             * @return true if export succeeded
             */
            bool export_csv(const std::string& filepath) const;

            /**
             * @brief Print a summary of the latest epoch metrics to stdout
             */
            void print_epoch_summary() const;

            /**
             * @brief Clear all logged data
             */
            void clear();

        private:
            int current_epoch_ = 0;
            // metric_name -> vector of (epoch, value) pairs
            std::map<std::string, std::vector<std::pair<int, float>>> history_;
        };
    }
}

#endif // VISUALIZATIONS_HPP
