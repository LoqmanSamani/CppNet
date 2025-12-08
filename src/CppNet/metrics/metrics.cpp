#include <cmath>
#include <algorithm>
#include "metrics/metrics.hpp"

namespace CppNet
{
    namespace Metrics
    {
        /************************************** Accuracy *************************************/
        Accuracy::Accuracy() : correct(0), total(0) {}

        double Accuracy::compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets)
        {
            // TODO: Implement: argmax(predictions) == targets
            return 0.0;
        }

        void Accuracy::reset()
        {
            correct = 0;
            total = 0;
        }

        void Accuracy::accumulate(double value, int n)
        {
            correct += static_cast<int>(value * n);
            total += n;
        }

        double Accuracy::result() const
        {
            return (total == 0) ? 0.0 : static_cast<double>(correct) / total;
        }

        /************************************** Top-K Accuracy *************************************/
        TopKAccuracy::TopKAccuracy(int k) : k(k), correct(0), total(0) {}

        double TopKAccuracy::compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets)
        {
            // TODO: Implement: check if target is in top-k predictions
            return 0.0;
        }

        void TopKAccuracy::reset()
        {
            correct = 0;
            total = 0;
        }

        void TopKAccuracy::accumulate(double value, int n)
        {
            correct += static_cast<int>(value * n);
            total += n;
        }

        double TopKAccuracy::result() const
        {
            return (total == 0) ? 0.0 : static_cast<double>(correct) / total;
        }

        /************************************** Precision *************************************/
        Precision::Precision() : true_positive(0), false_positive(0) {}

        double Precision::compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets)
        {
            // TODO: Implement: TP / (TP + FP)
            return 0.0;
        }

        void Precision::reset()
        {
            true_positive = 0;
            false_positive = 0;
        }

        void Precision::accumulate(double value, int n)
        {
            // Not batch-averaged here, will need true counts
        }

        double Precision::result() const
        {
            int denom = true_positive + false_positive;
            return (denom == 0) ? 0.0 : static_cast<double>(true_positive) / denom;
        }

        /************************************** Recall *************************************/
        Recall::Recall() : true_positive(0), false_negative(0) {}

        double Recall::compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets)
        {
            // TODO: Implement: TP / (TP + FN)
            return 0.0;
        }

        void Recall::reset()
        {
            true_positive = 0;
            false_negative = 0;
        }

        void Recall::accumulate(double value, int n)
        {
            // Not batch-averaged here, will need true counts
        }

        double Recall::result() const
        {
            int denom = true_positive + false_negative;
            return (denom == 0) ? 0.0 : static_cast<double>(true_positive) / denom;
        }

        /************************************** F1 Score *************************************/
        F1Score::F1Score() : precision(), recall() {}

        double F1Score::compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets)
        {
            // TODO: Implement: 2 * (P * R) / (P + R)
            return 0.0;
        }

        /************************************** MAE *************************************/
        MAE::MAE() {}

        double MAE::compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets)
        {
            // TODO: Implement MAE metric
            return 0.0;
        }

        double MAE::compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets)
        {
            return 0.0;
        }

        /************************************** MSE *************************************/
        MSE::MSE() {}

        double MSE::compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets)
        {
            // TODO: Implement MSE metric
            return 0.0;
        }

        double MSE::compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets)
        {
            return 0.0;
        }

        /************************************** R² Score *************************************/
        R2Score::R2Score() {}

        double R2Score::compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets)
        {
            // TODO: Implement R² = 1 - SS_res / SS_tot
            return 0.0;
        }

        double R2Score::compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets)
        {
            return 0.0;
        }
    }
}
