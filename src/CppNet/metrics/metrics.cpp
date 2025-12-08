#include <cmath>
#include <algorithm>
#include "CppNet/metrics/metrics.hpp"




namespace CppNet
{
    namespace Metrics
    {
        /************************************** Accuracy *************************************/
        Accuracy::Accuracy() : correct(0), total(0) {}

        float Accuracy::compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets)
        {
            // TODO: Implement: argmax(predictions) == targets
            return 0.0f;
        }

        void Accuracy::reset()
        {
            correct = 0;
            total = 0;
        }

        void Accuracy::accumulate(float value, int n)
        {
            correct += static_cast<int>(value * n);
            total += n;
        }

        float Accuracy::result() const
        {
            return (total == 0) ? 0.0f : static_cast<float>(correct) / total;
        }

        /************************************** Top-K Accuracy *************************************/
        TopKAccuracy::TopKAccuracy(int k) : k(k), correct(0), total(0) {}

        float TopKAccuracy::compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets)
        {
            // TODO: Implement: check if target is in top-k predictions
            return 0.0f;
        }

        void TopKAccuracy::reset()
        {
            correct = 0;
            total = 0;
        }

        void TopKAccuracy::accumulate(float value, int n)
        {
            correct += static_cast<int>(value * n);
            total += n;
        }

        float TopKAccuracy::result() const
        {
            return (total == 0) ? 0.0f : static_cast<float>(correct) / total;
        }

        /************************************** Precision *************************************/
        Precision::Precision() : true_positive(0), false_positive(0) {}

        float Precision::compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets)
        {
            // TODO: Implement: TP / (TP + FP)
            return 0.0f;
        }

        void Precision::reset()
        {
            true_positive = 0;
            false_positive = 0;
        }

        void Precision::accumulate(float value, int n)
        {
            // Not batch-averaged here, will need true counts
        }

        float Precision::result() const
        {
            int denom = true_positive + false_positive;
            return (denom == 0) ? 0.0f : static_cast<float>(true_positive) / denom;
        }

        /************************************** Recall *************************************/
        Recall::Recall() : true_positive(0), false_negative(0) {}

        float Recall::compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets)
        {
            // TODO: Implement: TP / (TP + FN)
            return 0.0f;
        }

        void Recall::reset()
        {
            true_positive = 0;
            false_negative = 0;
        }

        void Recall::accumulate(float value, int n)
        {
            // Not batch-averaged here, will need true counts
        }

        float Recall::result() const
        {
            int denom = true_positive + false_negative;
            return (denom == 0) ? 0.0f : static_cast<float>(true_positive) / denom;
        }

        /************************************** F1 Score *************************************/
        F1Score::F1Score() : precision(), recall() {}

        float F1Score::compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets)
        {
            // TODO: Implement: 2 * (P * R) / (P + R)
            return 0.0f;
        }

        /************************************** MAE *************************************/
        MAE::MAE() {}

        float MAE::compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets)
        {
            // TODO: Implement MAE metric
            return 0.0f;
        }

        float MAE::compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets)
        {
            return 0.0f;
        }

        /************************************** MSE *************************************/
        MSE::MSE() {}

        float MSE::compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets)
        {
            // TODO: Implement MSE metric
            return 0.0f;
        }

        float MSE::compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets)
        {
            return 0.0f;
        }

        /************************************** R² Score *************************************/
        R2Score::R2Score() {}

        float R2Score::compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets)
        {
            // TODO: Implement R² = 1 - SS_res / SS_tot
            return 0.0f;
        }

        float R2Score::compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets)
        {
            return 0.0f;
        }
    }
}
