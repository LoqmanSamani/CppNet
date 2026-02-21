/**
 * @file utils.hpp
 * @brief General utility functions for CppNet
 *
 * Provides data loading, tensor manipulation, normalization,
 * and one-hot encoding helpers.
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>
#include <utility>

namespace CppNet
{
    namespace Utils
    {
        /**
         * @brief Load a CSV file into a pair of feature and label tensors
         * @param filepath Path to the CSV file
         * @param label_col Index of the label column (-1 for last column)
         * @param has_header Whether the CSV has a header row
         * @return Pair of (features [N, D], labels [N, 1])
         */
        std::pair<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>
        load_csv(const std::string& filepath, int label_col = -1, bool has_header = true);

        /**
         * @brief One-hot encode integer labels
         * @param labels Integer labels [N]
         * @param num_classes Number of classes (auto-detect if -1)
         * @return One-hot tensor [N, num_classes]
         */
        Eigen::Tensor<float, 2> one_hot_encode(const std::vector<int>& labels, int num_classes = -1);

        /**
         * @brief Normalize features to zero mean and unit variance (column-wise)
         * @param data Input tensor [N, D]
         * @return Normalized tensor [N, D]
         */
        Eigen::Tensor<float, 2> normalize(const Eigen::Tensor<float, 2>& data);

        /**
         * @brief Min-max scale features to [0, 1] range (column-wise)
         * @param data Input tensor [N, D]
         * @return Scaled tensor [N, D]
         */
        Eigen::Tensor<float, 2> min_max_scale(const Eigen::Tensor<float, 2>& data);

        /**
         * @brief Shuffle data and labels in-place with the same permutation
         * @param data Feature tensor [N, D]
         * @param labels Label tensor [N, C]
         * @param seed Random seed (-1 for random)
         */
        void shuffle_data(Eigen::Tensor<float, 2>& data,
                          Eigen::Tensor<float, 2>& labels,
                          int seed = -1);

        /**
         * @brief Split data into train and validation sets
         * @param data Feature tensor [N, D]
         * @param labels Label tensor [N, C]
         * @param val_ratio Fraction of data for validation (0.0 to 1.0)
         * @return Tuple of (train_data, train_labels, val_data, val_labels)
         */
        std::tuple<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>,
                   Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>
        train_val_split(const Eigen::Tensor<float, 2>& data,
                        const Eigen::Tensor<float, 2>& labels,
                        float val_ratio = 0.2f);

        /**
         * @brief Get argmax indices along axis 1
         * @param tensor Input tensor [N, D]
         * @return Vector of indices of maximum values per row
         */
        std::vector<int> argmax(const Eigen::Tensor<float, 2>& tensor);
    }
}

#endif // UTILS_HPP
