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
            virtual void apply(Eigen::Tensor<float, 2>& params) = 0;

            // Return penalty term for loss
            virtual float penalty(const Eigen::Tensor<float, 2>& params) = 0;
        };

        /************************************** L1 Regularization *************************************/
        class L1 : public Regularizer
        {
        private:
            float lambda;

        public:
            L1(float lambda = 0.01f);
            void apply(Eigen::Tensor<float, 2>& params) override;
            float penalty(const Eigen::Tensor<float, 2>& params) override;
        };

        /************************************** L2 Regularization *************************************/
        class L2 : public Regularizer
        {
        private:
            float lambda;

        public:
            L2(float lambda = 0.01);
            void apply(Eigen::Tensor<float, 2>& params) override;
            float penalty(const Eigen::Tensor<float, 2>& params) override;
        };

        /************************************** ElasticNet (L1 + L2) *************************************/
        class ElasticNet : public Regularizer
        {
        private:
            float l1;
            float l2;

        public:
            ElasticNet(float l1 = 0.01, float l2 = 0.01);
            void apply(Eigen::Tensor<float, 2>& params) override;
            float penalty(const Eigen::Tensor<float, 2>& params) override;
        };

        /************************************** Dropout *************************************/
        class Dropout : public Regularizer
        {
        private:
            float rate;

        public:
            Dropout(float rate = 0.5);
            void apply(Eigen::Tensor<float, 2>& params) override;
            float penalty(const Eigen::Tensor<float, 2>& params) override;
        };
    }
}

#endif // REGULARIZATIONS_HPP
