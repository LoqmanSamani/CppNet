/**
 * @file metrics.cpp
 * @brief Implementation of evaluation metrics
 */

#include "CppNet/metrics/metrics.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace CppNet
{
    namespace Metrics
    {
        float accuracy(const Eigen::Tensor<float, 2>& predictions,
                       const Eigen::Tensor<float, 2>& targets)
        {
            int batch_size = predictions.dimension(0);
            int num_classes = predictions.dimension(1);

            if (batch_size != targets.dimension(0) || num_classes != targets.dimension(1))
            {
                throw std::invalid_argument("Predictions and targets shape mismatch");
            }

            int correct = 0;

            #ifdef USE_OPENMP
            #pragma omp parallel for reduction(+:correct)
            #endif
            for (int i = 0; i < batch_size; ++i)
            {
                int pred_class = 0;
                int target_class = 0;
                float pred_max = predictions(i, 0);
                float target_max = targets(i, 0);

                for (int j = 1; j < num_classes; ++j)
                {
                    if (predictions(i, j) > pred_max)
                    {
                        pred_max = predictions(i, j);
                        pred_class = j;
                    }
                    if (targets(i, j) > target_max)
                    {
                        target_max = targets(i, j);
                        target_class = j;
                    }
                }

                if (pred_class == target_class)
                    ++correct;
            }

            return static_cast<float>(correct) / static_cast<float>(batch_size);
        }

        float binary_accuracy(const Eigen::Tensor<float, 2>& predictions,
                              const Eigen::Tensor<float, 2>& targets,
                              float threshold)
        {
            int batch_size = predictions.dimension(0);
            int correct = 0;

            #ifdef USE_OPENMP
            #pragma omp parallel for reduction(+:correct)
            #endif
            for (int i = 0; i < batch_size; ++i)
            {
                int pred_label = (predictions(i, 0) >= threshold) ? 1 : 0;
                int target_label = (targets(i, 0) >= 0.5f) ? 1 : 0;
                if (pred_label == target_label)
                    ++correct;
            }

            return static_cast<float>(correct) / static_cast<float>(batch_size);
        }

        float precision(const Eigen::Tensor<float, 2>& predictions,
                        const Eigen::Tensor<float, 2>& targets,
                        float threshold)
        {
            int batch_size = predictions.dimension(0);
            int true_positive = 0;
            int false_positive = 0;

            for (int i = 0; i < batch_size; ++i)
            {
                int pred_label = (predictions(i, 0) >= threshold) ? 1 : 0;
                int target_label = (targets(i, 0) >= 0.5f) ? 1 : 0;

                if (pred_label == 1 && target_label == 1)
                    ++true_positive;
                else if (pred_label == 1 && target_label == 0)
                    ++false_positive;
            }

            int denominator = true_positive + false_positive;
            if (denominator == 0)
                return 0.0f;

            return static_cast<float>(true_positive) / static_cast<float>(denominator);
        }

        float recall(const Eigen::Tensor<float, 2>& predictions,
                     const Eigen::Tensor<float, 2>& targets,
                     float threshold)
        {
            int batch_size = predictions.dimension(0);
            int true_positive = 0;
            int false_negative = 0;

            for (int i = 0; i < batch_size; ++i)
            {
                int pred_label = (predictions(i, 0) >= threshold) ? 1 : 0;
                int target_label = (targets(i, 0) >= 0.5f) ? 1 : 0;

                if (pred_label == 1 && target_label == 1)
                    ++true_positive;
                else if (pred_label == 0 && target_label == 1)
                    ++false_negative;
            }

            int denominator = true_positive + false_negative;
            if (denominator == 0)
                return 0.0f;

            return static_cast<float>(true_positive) / static_cast<float>(denominator);
        }

        float f1_score(const Eigen::Tensor<float, 2>& predictions,
                       const Eigen::Tensor<float, 2>& targets,
                       float threshold)
        {
            float p = precision(predictions, targets, threshold);
            float r = recall(predictions, targets, threshold);

            if (p + r == 0.0f)
                return 0.0f;

            return 2.0f * (p * r) / (p + r);
        }
    }
}
