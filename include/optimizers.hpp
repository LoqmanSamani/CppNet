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
                virtual void update(Layers::Linear& layer, double learning_rate) = 0;
                virtual ~Optimizer() = default;
        };

        class SGD : public Optimizer 
        {
        public:
            void update(Layers::Linear& layer, double learning_rate) override;
        };

    }
}

#endif // OPTIMIZERS_HPP
