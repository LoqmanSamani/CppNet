#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>
#include <memory>
#include "layers/linear.hpp"




namespace CppNet
{
    namespace Optimizers
    {
        class Optimizer
        {
        public:
            virtual void step(CppNet::Layers::Linear& layer, double learning_rate) = 0;
            virtual void step(CppNet::Layers::Conv2d& layer, double learning_rate) = 0;
            virtual void step(CppNet::Layers::MultiHeadAttention& layer, double learning_rate) = 0;
            virtual ~Optimizer() = default;
        };
        
        /************************************** SGD *************************************/
        class SGD : public Optimizer
        {
        public:
            SGD() = default; // Explicit default constructor
            void step(CppNet::Layers::Linear& layer, double learning_rate) override;
            void step(CppNet::Layers::Conv2d& layer, double learning_rate) override;
            void step(CppNet::Layers::MultiHeadAttention& layer, double learning_rate) override;
        };

        
        /************************************** Momentum *************************************/
        //class Momentum : public Optimizer
        //{
        //private:
        //    double lr;
        //    double momentum;

        //public:
        //    Momentum(double lr = 0.01, double momentum = 0.9);
        //    void step(CppNet::Layers::Linear& layer, double learning_rate) override;
        //};

        /************************************** Adam *************************************/
        //class Adam : public Optimizer
        //{
        //private:
        //    double lr;
        //    double beta1;
        //    double beta2;
        //    double eps;

        //public:
        //    Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
        //    void step(CppNet::Layers::Linear& layer, double learning_rate) override;
        //};

        /************************************** RMSProp *************************************/
        //class RMSProp : public Optimizer
        //{
        //private:
        //    double lr;
        //    double rho;
        //   double eps;
        //public:
        //    RMSProp(double lr = 0.001, double rho = 0.9, double eps = 1e-8);
        //    void step(CppNet::Layers::Linear& layer, double learning_rate) override;
        //};

        /************************************** Adagrad *************************************/
        //class Adagrad : public Optimizer
        //{
        //private:
        //    double lr;
        //    double eps;

        //public:
        //    Adagrad(double lr = 0.01, double eps = 1e-8);
        //    void step(CppNet::Layers::Linear& layer, double learning_rate) override;
        //};
        
    }
}

#endif // OPTIMIZERS_HPP