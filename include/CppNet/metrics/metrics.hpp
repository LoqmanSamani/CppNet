/**
 * @file metrics.hpp
 * @brief Evaluation metrics for CppNet
 *
 * Provides accuracy, precision, recall, and F1 score computations
 * for classification tasks.
 */

#ifndef METRICS_HPP
#define METRICS_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>

namespace CppNet
{
    namespace Metrics
    {
        /**
         * @brief Compute classification accuracy
         * @param predictions Tensor of predicted class probabilities [batch, num_classes]
         * @param targets Tensor of true labels (one-hot) [batch, num_classes]
         * @return Accuracy in range [0, 1]
         */
        float accuracy(const Eigen::Tensor<float, 2>& predictions,
                       const Eigen::Tensor<float, 2>& targets);

        /**
         * @brief Compute binary accuracy from raw predictions
         * @param predictions Tensor of predicted probabilities [batch, 1]
         * @param targets Tensor of true binary labels [batch, 1]
         * @param threshold Decision threshold (default 0.5)
         * @return Binary accuracy in range [0, 1]
         */
        float binary_accuracy(const Eigen::Tensor<float, 2>& predictions,
                              const Eigen::Tensor<float, 2>& targets,
                              float threshold = 0.5f);

        /**
         * @brief Compute precision for binary classification
         * @param predictions Predicted probabilities [batch, 1]
         * @param targets True labels [batch, 1]
         * @param threshold Decision threshold
         * @return Precision score
         */
        float precision(const Eigen::Tensor<float, 2>& predictions,
                        const Eigen::Tensor<float, 2>& targets,
                        float threshold = 0.5f);

        /**
         * @brief Compute recall for binary classification
         * @param predictions Predicted probabilities [batch, 1]
         * @param targets True labels [batch, 1]
         * @param threshold Decision threshold
         * @return Recall score
         */
        float recall(const Eigen::Tensor<float, 2>& predictions,
                     const Eigen::Tensor<float, 2>& targets,
                     float threshold = 0.5f);

        /**
         * @brief Compute F1 score for binary classification
         * @param predictions Predicted probabilities [batch, 1]
         * @param targets True labels [batch, 1]
         * @param threshold Decision threshold
         * @return F1 score
         */
        float f1_score(const Eigen::Tensor<float, 2>& predictions,
                       const Eigen::Tensor<float, 2>& targets,
                       float threshold = 0.5f);
    }
}

#endif // METRICS_HPP
