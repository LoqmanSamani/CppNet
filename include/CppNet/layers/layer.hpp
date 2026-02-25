#ifndef LAYER_HPP
#define LAYER_HPP



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

            /// @brief Reset accumulated gradients (default: no-op for non-trainable layers)
            virtual void reset_grads() {}

            virtual ~Layer() = default;
        };
    }
}

#endif // LAYER_HPP