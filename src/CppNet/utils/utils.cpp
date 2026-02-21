/**
 * @file utils.cpp
 * @brief Implementation of general utility functions
 */

#include "CppNet/utils/utils.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace CppNet
{
    namespace Utils
    {
        std::pair<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>
        load_csv(const std::string& filepath, int label_col, bool has_header)
        {
            std::ifstream file(filepath);
            if (!file.is_open())
            {
                throw std::runtime_error("Cannot open file: " + filepath);
            }

            std::vector<std::vector<float>> rows;
            std::string line;

            // Skip header if present
            if (has_header && std::getline(file, line))
            {
                // header line consumed
            }

            while (std::getline(file, line))
            {
                std::vector<float> row;
                std::stringstream ss(line);
                std::string cell;

                while (std::getline(ss, cell, ','))
                {
                    try
                    {
                        row.push_back(std::stof(cell));
                    }
                    catch (...)
                    {
                        row.push_back(0.0f); // default for non-numeric
                    }
                }

                if (!row.empty())
                    rows.push_back(row);
            }

            if (rows.empty())
            {
                throw std::runtime_error("CSV file is empty: " + filepath);
            }

            int num_samples = static_cast<int>(rows.size());
            int num_cols = static_cast<int>(rows[0].size());

            // Resolve label column (-1 means last)
            if (label_col < 0)
                label_col = num_cols + label_col;

            if (label_col < 0 || label_col >= num_cols)
            {
                throw std::invalid_argument("Invalid label column index");
            }

            int num_features = num_cols - 1;

            Eigen::Tensor<float, 2> features(num_samples, num_features);
            Eigen::Tensor<float, 2> labels(num_samples, 1);

            for (int i = 0; i < num_samples; ++i)
            {
                int feat_idx = 0;
                for (int j = 0; j < num_cols; ++j)
                {
                    if (j == label_col)
                    {
                        labels(i, 0) = rows[i][j];
                    }
                    else
                    {
                        features(i, feat_idx) = rows[i][j];
                        ++feat_idx;
                    }
                }
            }

            return {features, labels};
        }

        Eigen::Tensor<float, 2> one_hot_encode(const std::vector<int>& labels, int num_classes)
        {
            if (labels.empty())
            {
                throw std::invalid_argument("Labels vector is empty");
            }

            if (num_classes < 0)
            {
                num_classes = *std::max_element(labels.begin(), labels.end()) + 1;
            }

            int num_samples = static_cast<int>(labels.size());
            Eigen::Tensor<float, 2> one_hot(num_samples, num_classes);
            one_hot.setZero();

            for (int i = 0; i < num_samples; ++i)
            {
                if (labels[i] >= 0 && labels[i] < num_classes)
                {
                    one_hot(i, labels[i]) = 1.0f;
                }
            }

            return one_hot;
        }

        Eigen::Tensor<float, 2> normalize(const Eigen::Tensor<float, 2>& data)
        {
            int rows = data.dimension(0);
            int cols = data.dimension(1);

            Eigen::Tensor<float, 2> result(rows, cols);

            for (int j = 0; j < cols; ++j)
            {
                // Compute column mean
                float sum = 0.0f;
                for (int i = 0; i < rows; ++i)
                    sum += data(i, j);
                float mean = sum / static_cast<float>(rows);

                // Compute column std
                float sq_sum = 0.0f;
                for (int i = 0; i < rows; ++i)
                {
                    float diff = data(i, j) - mean;
                    sq_sum += diff * diff;
                }
                float stddev = std::sqrt(sq_sum / static_cast<float>(rows));

                // Avoid division by zero
                if (stddev < 1e-8f)
                    stddev = 1e-8f;

                for (int i = 0; i < rows; ++i)
                {
                    result(i, j) = (data(i, j) - mean) / stddev;
                }
            }

            return result;
        }

        Eigen::Tensor<float, 2> min_max_scale(const Eigen::Tensor<float, 2>& data)
        {
            int rows = data.dimension(0);
            int cols = data.dimension(1);

            Eigen::Tensor<float, 2> result(rows, cols);

            for (int j = 0; j < cols; ++j)
            {
                float col_min = data(0, j);
                float col_max = data(0, j);

                for (int i = 1; i < rows; ++i)
                {
                    col_min = std::min(col_min, data(i, j));
                    col_max = std::max(col_max, data(i, j));
                }

                float range = col_max - col_min;
                if (range < 1e-8f)
                    range = 1e-8f;

                for (int i = 0; i < rows; ++i)
                {
                    result(i, j) = (data(i, j) - col_min) / range;
                }
            }

            return result;
        }

        void shuffle_data(Eigen::Tensor<float, 2>& data,
                          Eigen::Tensor<float, 2>& labels,
                          int seed)
        {
            int num_samples = data.dimension(0);
            int num_features = data.dimension(1);
            int num_label_cols = labels.dimension(1);

            // Create index permutation
            std::vector<int> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);

            std::mt19937 rng;
            if (seed >= 0)
                rng.seed(static_cast<unsigned>(seed));
            else
                rng.seed(std::random_device{}());

            std::shuffle(indices.begin(), indices.end(), rng);

            // Apply permutation
            Eigen::Tensor<float, 2> shuffled_data(num_samples, num_features);
            Eigen::Tensor<float, 2> shuffled_labels(num_samples, num_label_cols);

            for (int i = 0; i < num_samples; ++i)
            {
                for (int j = 0; j < num_features; ++j)
                    shuffled_data(i, j) = data(indices[i], j);
                for (int j = 0; j < num_label_cols; ++j)
                    shuffled_labels(i, j) = labels(indices[i], j);
            }

            data = shuffled_data;
            labels = shuffled_labels;
        }

        std::tuple<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>,
                   Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>
        train_val_split(const Eigen::Tensor<float, 2>& data,
                        const Eigen::Tensor<float, 2>& labels,
                        float val_ratio)
        {
            int num_samples = data.dimension(0);
            int num_features = data.dimension(1);
            int num_label_cols = labels.dimension(1);

            int val_size = static_cast<int>(num_samples * val_ratio);
            int train_size = num_samples - val_size;

            Eigen::Tensor<float, 2> train_data(train_size, num_features);
            Eigen::Tensor<float, 2> train_labels(train_size, num_label_cols);
            Eigen::Tensor<float, 2> val_data(val_size, num_features);
            Eigen::Tensor<float, 2> val_labels(val_size, num_label_cols);

            for (int i = 0; i < train_size; ++i)
            {
                for (int j = 0; j < num_features; ++j)
                    train_data(i, j) = data(i, j);
                for (int j = 0; j < num_label_cols; ++j)
                    train_labels(i, j) = labels(i, j);
            }

            for (int i = 0; i < val_size; ++i)
            {
                for (int j = 0; j < num_features; ++j)
                    val_data(i, j) = data(train_size + i, j);
                for (int j = 0; j < num_label_cols; ++j)
                    val_labels(i, j) = labels(train_size + i, j);
            }

            return {train_data, train_labels, val_data, val_labels};
        }

        std::vector<int> argmax(const Eigen::Tensor<float, 2>& tensor)
        {
            int rows = tensor.dimension(0);
            int cols = tensor.dimension(1);

            std::vector<int> result(rows);

            for (int i = 0; i < rows; ++i)
            {
                int max_idx = 0;
                float max_val = tensor(i, 0);
                for (int j = 1; j < cols; ++j)
                {
                    if (tensor(i, j) > max_val)
                    {
                        max_val = tensor(i, j);
                        max_idx = j;
                    }
                }
                result[i] = max_idx;
            }

            return result;
        }
    }
}
