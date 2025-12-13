#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "CppNet/layers/linear.hpp"

namespace CppNet
{
    namespace Optimizers
    {
        class Optimizer
        {
        public:
            virtual void step(CppNet::Layers::Linear& layer, float learning_rate) = 0;
            //virtual void step(CppNet::Layers::Conv2d& layer, float learning_rate) = 0;
            //virtual void step(CppNet::Layers::MultiHeadAttention& layer, float learning_rate) = 0;
            virtual ~Optimizer() = default;
        };
    }
}

#endif // OPTIMIZER_HPP