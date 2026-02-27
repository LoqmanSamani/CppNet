/**
 * @file visualizations.cpp
 * @brief Implementation of TrainingLogger
 */

#include "CppNet/visualizations/visualizations.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace CppNet
{
    namespace Visualizations
    {
        void TrainingLogger::log(const std::string& name, float value)
        {
            history_[name].push_back({current_epoch_, value});
        }

        void TrainingLogger::next_epoch()
        {
            ++current_epoch_;
        }

        std::vector<std::pair<int, float>> TrainingLogger::get_history(const std::string& name) const
        {
            auto it = history_.find(name);
            if (it != history_.end())
                return it->second;
            return {};
        }

        std::vector<std::string> TrainingLogger::get_metric_names() const
        {
            std::vector<std::string> names;
            names.reserve(history_.size());
            for (const auto& entry : history_)
                names.push_back(entry.first);
            return names;
        }

        bool TrainingLogger::export_csv(const std::string& filepath) const
        {
            std::ofstream file(filepath);
            if (!file.is_open())
                return false;

            // collect all metric names
            auto names = get_metric_names();
            if (names.empty())
                return false;

            // determine max epoch
            int max_epoch = 0;
            for (const auto& entry : history_)
            {
                for (const auto& pair : entry.second)
                {
                    max_epoch = std::max(max_epoch, pair.first);
                }
            }

            // write header
            file << "epoch";
            for (const auto& name : names)
                file << "," << name;
            file << "\n";

            // write data rows
            for (int epoch = 0; epoch <= max_epoch; ++epoch)
            {
                file << epoch;
                for (const auto& name : names)
                {
                    file << ",";
                    auto it = history_.find(name);
                    if (it != history_.end())
                    {
                        bool found = false;
                        for (const auto& pair : it->second)
                        {
                            if (pair.first == epoch)
                            {
                                file << std::fixed << std::setprecision(6) << pair.second;
                                found = true;
                                break;
                            }
                        }
                        if (!found)
                            file << "";
                    }
                }
                file << "\n";
            }

            file.close();
            return true;
        }

        void TrainingLogger::print_epoch_summary() const
        {
            std::cout << "Epoch " << current_epoch_ << " | ";
            for (const auto& entry : history_)
            {
                if (!entry.second.empty() && entry.second.back().first == current_epoch_)
                {
                    std::cout << entry.first << ": "
                              << std::fixed << std::setprecision(4)
                              << entry.second.back().second << " | ";
                }
            }
            std::cout << std::endl;
        }

        void TrainingLogger::clear()
        {
            history_.clear();
            current_epoch_ = 0;
        }
    }
}
