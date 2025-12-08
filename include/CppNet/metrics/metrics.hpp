#ifndef METRICS_HPP
#define METRICS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <vector>

namespace CppNet
{
    namespace Metrics
    {
        class Metric
        {
        public:
            virtual ~Metric() = default;

            // Compute metric on batch
            virtual float compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets) { return 0.0f; }
            virtual float compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets) { return 0.0f; }
            virtual float compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets) { return 0.0f; }

            // For streaming metrics (accumulate across batches)
            virtual void reset() {}
            virtual void accumulate(float value, int n = 1) {}
            virtual float result() const { return 0.0f; }
        };

        /************************************** Accuracy *************************************/
        class Accuracy : public Metric
        {
        private:
            int correct;
            int total;

        public:
            Accuracy();
            float compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets) override;

            void reset() override;
            void accumulate(float value, int n = 1) override;
            float result() const override;
        };

        /************************************** Top-K Accuracy *************************************/
        class TopKAccuracy : public Metric
        {
        private:
            int k;
            int correct;
            int total;

        public:
            TopKAccuracy(int k = 5);
            float compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<int, 1>& targets) override;

            void reset() override;
            void accumulate(float value, int n = 1) override;
            float result() const override;
        };

        /************************************** Precision *************************************/
        class Precision : public Metric
        {
        private:
            int true_positive;
            int false_positive;

        public:
            Precision();
            float compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets) override;

            void reset() override;
            void accumulate(float value, int n = 1) override;
            float result() const override;
        };

        /************************************** Recall *************************************/
        class Recall : public Metric
        {
        private:
            int true_positive;
            int false_negative;

        public:
            Recall();
            float compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets) override;

            void reset() override;
            void accumulate(float value, int n = 1) override;
            float result() const override;
        };

        /************************************** F1 Score *************************************/
        class F1Score : public Metric
        {
        private:
            Precision precision;
            Recall recall;

        public:
            F1Score();
            float compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets) override;
        };

        /************************************** Mean Absolute Error (MAE) *************************************/
        class MAE : public Metric
        {
        public:
            MAE();
            float compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets) override;
            float compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets) override;
        };

        /************************************** Mean Squared Error (MSE) *************************************/
        class MSE : public Metric
        {
        public:
            MSE();
            float compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets) override;
            float compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets) override;
        };

        /************************************** R² Score *************************************/
        class R2Score : public Metric
        {
        public:
            R2Score();
            float compute(const Eigen::Tensor<float, 1>& predictions, const Eigen::Tensor<float, 1>& targets) override;
            float compute(const Eigen::Tensor<float, 2>& predictions, const Eigen::Tensor<float, 2>& targets) override;
        };
    }
}

#endif // METRICS_HPP