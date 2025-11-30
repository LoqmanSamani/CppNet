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
            virtual double compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) { return 0.0; }
            virtual double compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets) { return 0.0; }
            virtual double compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) { return 0.0; }

            // For streaming metrics (accumulate across batches)
            virtual void reset() {}
            virtual void accumulate(double value, int n = 1) {}
            virtual double result() const { return 0.0; }
        };

        /************************************** Accuracy *************************************/
        class Accuracy : public Metric
        {
        private:
            int correct;
            int total;

        public:
            Accuracy();
            double compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets) override;

            void reset() override;
            void accumulate(double value, int n = 1) override;
            double result() const override;
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
            double compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<int, 1>& targets) override;

            void reset() override;
            void accumulate(double value, int n = 1) override;
            double result() const override;
        };

        /************************************** Precision *************************************/
        class Precision : public Metric
        {
        private:
            int true_positive;
            int false_positive;

        public:
            Precision();
            double compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) override;

            void reset() override;
            void accumulate(double value, int n = 1) override;
            double result() const override;
        };

        /************************************** Recall *************************************/
        class Recall : public Metric
        {
        private:
            int true_positive;
            int false_negative;

        public:
            Recall();
            double compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) override;

            void reset() override;
            void accumulate(double value, int n = 1) override;
            double result() const override;
        };

        /************************************** F1 Score *************************************/
        class F1Score : public Metric
        {
        private:
            Precision precision;
            Recall recall;

        public:
            F1Score();
            double compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) override;
        };

        /************************************** Mean Absolute Error (MAE) *************************************/
        class MAE : public Metric
        {
        public:
            MAE();
            double compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) override;
            double compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) override;
        };

        /************************************** Mean Squared Error (MSE) *************************************/
        class MSE : public Metric
        {
        public:
            MSE();
            double compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) override;
            double compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) override;
        };

        /************************************** R² Score *************************************/
        class R2Score : public Metric
        {
        public:
            R2Score();
            double compute(const Eigen::Tensor<double, 1>& predictions, const Eigen::Tensor<double, 1>& targets) override;
            double compute(const Eigen::Tensor<double, 2>& predictions, const Eigen::Tensor<double, 2>& targets) override;
        };
    }
}

#endif // METRICS_HPP
