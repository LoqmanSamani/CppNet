#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "layers.hpp"

namespace CppNet {
    
    class Optimizer {
        public:
            virtual void update(Linear& layer, double learning_rate) = 0;
            virtual ~Optimizer() = default;
    };

    class SGD : public Optimizer {
    public:
        void update(Linear& layer, double learning_rate) override;
    };
}

#endif // OPTIMIZERS_HPP
