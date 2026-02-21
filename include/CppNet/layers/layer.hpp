#ifndef LAYER_HPP
#define LAYER_HPP



// Base Layer class definition

namespace CppNet
{
    namespace Optimizers
    {
        class Optimizer;
    }

    namespace Layers
    {
        class Layer
        {
        public:
            virtual bool is_trainable() const = 0;
            virtual void step(Optimizers::Optimizer& optimizer, float learning_rate) = 0;
            virtual ~Layer() = default;
        };
    }
}

#endif // LAYER_HPP