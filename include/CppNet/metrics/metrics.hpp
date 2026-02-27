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
        float accuracy(const Eigen::Tensor<float, 2>& predictions,
                       const Eigen::Tensor<float, 2>& targets);

        float binary_accuracy(const Eigen::Tensor<float, 2>& predictions,
                              const Eigen::Tensor<float, 2>& targets,
                              float threshold = 0.5f);

        float precision(const Eigen::Tensor<float, 2>& predictions,
                        const Eigen::Tensor<float, 2>& targets,
                        float threshold = 0.5f);
       
        float recall(const Eigen::Tensor<float, 2>& predictions,
                     const Eigen::Tensor<float, 2>& targets,
                     float threshold = 0.5f);

        float f1_score(const Eigen::Tensor<float, 2>& predictions,
                       const Eigen::Tensor<float, 2>& targets,
                       float threshold = 0.5f);
    }
}

#endif // METRICS_HPP
