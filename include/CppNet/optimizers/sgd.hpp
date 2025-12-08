#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>
#include <memory>
#include "CppNet/layers/linear.hpp"




namespace CppNet
{
    namespace Optimizers
    {
        class Optimizer
        {
        public:
            virtual void step(CppNet::Layers::Linear& layer, double learning_rate) = 0;
            //virtual void step(CppNet::Layers::Conv2d& layer, double learning_rate) = 0;
            //virtual void step(CppNet::Layers::MultiHeadAttention& layer, double learning_rate) = 0;
            virtual ~Optimizer() = default;
        };
        
        /************************************** SGD *************************************/
        class SGD : public Optimizer
        {
        public:
            SGD() = default; // explicit default constructor
            void step(CppNet::Layers::Linear& layer, double learning_rate) override;
            //void step(CppNet::Layers::Conv2d& layer, double learning_rate) override;
            //void step(CppNet::Layers::MultiHeadAttention& layer, double learning_rate) override;
        };
    }
}

#endif // OPTIMIZERS_HPP