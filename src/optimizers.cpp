#include "optimizers.hpp"



namespace CppNet
{
    namespace Optimizers
    {
        void SGD::update(CppNet::Layers::Linear& layer, double learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }

            // validate dimensions
            const auto& weights = layer.get_weights();
            const auto& grad_weights = layer.get_grad_weights();
            if (weights.dimensions() != grad_weights.dimensions())
            {
                throw std::runtime_error("Weight and gradient dimension mismatch in Linear layer: " + layer.get_layer_name());
            }

            layer.get_weights() -= learning_rate * layer.get_grad_weights();

            if (layer.has_bias())
            {
                const auto& biases = layer.get_biases();
                const auto& grad_biases = layer.get_grad_biases();
                if (biases.dimensions() != grad_biases.dimensions())
                {
                    throw std::runtime_error("Bias and gradient dimension mismatch in Linear layer: " + layer.get_layer_name());
                }
                layer.get_biases() -= learning_rate * layer.get_grad_biases();
            }
        }

        void SGD::update(CppNet::Layers::Conv2d& layer, double learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }

            // validate dimensions
            const auto& weights = layer.get_weights();
            const auto& grad_weights = layer.get_grad_weights();
            if (weights.dimensions() != grad_weights.dimensions())
            {
                throw std::runtime_error("Weight and gradient dimension mismatch in Conv2d layer: " + layer.get_layer_name());
            }

            layer.get_weights() -= learning_rate * layer.get_grad_weights();

            if (layer.has_bias())
            {
                const auto& biases = layer.get_biases();
                const auto& grad_biases = layer.get_grad_biases();
                if (biases.dimensions() != grad_biases.dimensions())
                {
                    throw std::runtime_error("Bias and gradient dimension mismatch in Conv2d layer: " + layer.get_layer_name());
                }
                layer.get_biases() -= learning_rate * layer.get_grad_biases();
            }
        }
    }
}