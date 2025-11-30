#ifndef REGULARIZATIONS_HPP
#define REGULARIZATIONS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>
#include <vector>

namespace CppNet
{
    namespace Regularizations
    {
        class Regularizer
        {
        public:
            virtual ~Regularizer() = default;

            // Apply regularization update to parameters (stub)
            virtual void apply(Eigen::Tensor<double, 2>& params) = 0;

            // Return penalty term for loss
            virtual double penalty(const Eigen::Tensor<double, 2>& params) = 0;
        };

        /************************************** L1 Regularization *************************************/
        class L1 : public Regularizer
        {
        private:
            double lambda;

        public:
            L1(double lambda = 0.01);
            void apply(Eigen::Tensor<double, 2>& params) override;
            double penalty(const Eigen::Tensor<double, 2>& params) override;
        };

        /************************************** L2 Regularization *************************************/
        class L2 : public Regularizer
        {
        private:
            double lambda;

        public:
            L2(double lambda = 0.01);
            void apply(Eigen::Tensor<double, 2>& params) override;
            double penalty(const Eigen::Tensor<double, 2>& params) override;
        };

        /************************************** ElasticNet (L1 + L2) *************************************/
        class ElasticNet : public Regularizer
        {
        private:
            double l1;
            double l2;

        public:
            ElasticNet(double l1 = 0.01, double l2 = 0.01);
            void apply(Eigen::Tensor<double, 2>& params) override;
            double penalty(const Eigen::Tensor<double, 2>& params) override;
        };

        /************************************** Dropout *************************************/
        class Dropout : public Regularizer
        {
        private:
            double rate;

        public:
            Dropout(double rate = 0.5);
            void apply(Eigen::Tensor<double, 2>& params) override;
            double penalty(const Eigen::Tensor<double, 2>& params) override;
        };
    }
}

#endif // REGULARIZATIONS_HPP
