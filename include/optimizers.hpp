#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>
#include <memory>

namespace CppNet
{
    namespace Optimizers
    {
        class Optimizer
        {
        public:
            virtual ~Optimizer() = default;

            // Apply parameter update (stub)
            virtual void step(std::vector<Eigen::Tensor<double, 2>>& params,
                              std::vector<Eigen::Tensor<double, 2>>& grads) = 0;

            // Reset gradients
            virtual void zero_grad(std::vector<Eigen::Tensor<double, 2>>& grads);
        };

        /************************************** SGD *************************************/
        class SGD : public Optimizer
        {
        private:
            double lr;      // Learning rate

        public:
            SGD(double lr = 0.01);
            void step(std::vector<Eigen::Tensor<double, 2>>& params,
                      std::vector<Eigen::Tensor<double, 2>>& grads) override;
        };

        /************************************** Momentum *************************************/
        class Momentum : public Optimizer
        {
        private:
            double lr;
            double momentum;

        public:
            Momentum(double lr = 0.01, double momentum = 0.9);
            void step(std::vector<Eigen::Tensor<double, 2>>& params,
                      std::vector<Eigen::Tensor<double, 2>>& grads) override;
        };

        /************************************** Adam *************************************/
        class Adam : public Optimizer
        {
        private:
            double lr;
            double beta1;
            double beta2;
            double eps;

        public:
            Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
            void step(std::vector<Eigen::Tensor<double, 2>>& params,
                      std::vector<Eigen::Tensor<double, 2>>& grads) override;
        };

        /************************************** RMSProp *************************************/
        class RMSProp : public Optimizer
        {
        private:
            double lr;
            double rho;
            double eps;

        public:
            RMSProp(double lr = 0.001, double rho = 0.9, double eps = 1e-8);
            void step(std::vector<Eigen::Tensor<double, 2>>& params,
                      std::vector<Eigen::Tensor<double, 2>>& grads) override;
        };

        /************************************** Adagrad *************************************/
        class Adagrad : public Optimizer
        {
        private:
            double lr;
            double eps;

        public:
            Adagrad(double lr = 0.01, double eps = 1e-8);
            void step(std::vector<Eigen::Tensor<double, 2>>& params,
                      std::vector<Eigen::Tensor<double, 2>>& grads) override;
        };
    }
}

#endif // OPTIMIZERS_HPP