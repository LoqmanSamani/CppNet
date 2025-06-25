#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "layers.hpp"

namespace CppNet
{
    namespace Optimizers
    {
       
        class Optimizer
        {
        public:
            virtual void update(CppNet::Layers::Linear& layer, double learning_rate) = 0;
            virtual void update(CppNet::Layers::Conv2d& layer, double learning_rate) = 0;
            virtual void update(CppNet::Layers::MultiHeadAttention& layer, double learning_rate) = 0;
            virtual ~Optimizer() = default;
        };

        
        class SGD : public Optimizer
        {
        public:
            SGD() = default; // Explicit default constructor
            void update(CppNet::Layers::Linear& layer, double learning_rate) override;
            void update(CppNet::Layers::Conv2d& layer, double learning_rate) override;
            void update(CppNet::Layers::MultiHeadAttention& layer, double learning_rate) override;
        };
    }
}

#endif // OPTIMIZERS_HPP