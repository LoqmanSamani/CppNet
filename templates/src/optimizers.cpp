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

        void SGD::update(CppNet::Layers::MultiHeadAttention& layer, double learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }
            
            // validate dimensions
            const auto& query_weights = layer.get_query_weights();
            const auto& key_weights = layer.get_key_weights();
            const auto& val_weights = layer.get_value_weights();

            const auto& grad_query_weights = layer.get_grad_query_weights();
            const auto& grad_key_weights = layer.get_grad_key_weights();
            const auto& grad_val_weights = layer.get_grad_value_weights();

            if (
                query_weights.dimensions() != grad_query_weights.dimensions() || 
                key_weights.dimensions() != grad_key_weights.dimensions() || 
                val_weights.dimensions() != grad_val_weights.dimensions())
            {
                throw std::runtime_error("Weight and gradient dimension mismatch in multi-head attention layer: " + layer.get_layer_name());
            }

            layer.get_query_weights() -= learning_rate * layer.get_grad_query_weights();
            layer.get_key_weights() -= learning_rate * layer.get_grad_key_weights();
            layer.get_value_weights() -= learning_rate * layer.get_grad_value_weights();

            if (layer.has_bias())
            {
                const auto& query_biases = layer.get_query_biases();
                const auto& key_biases = layer.get_key_biases();
                const auto& val_biases = layer.get_value_biases();

                const auto& grad_query_biases = layer.get_grad_query_biases();
                const auto& grad_key_biases = layer.get_grad_key_biases();
                const auto& grad_val_biases = layer.get_grad_value_biases();

                if (
                    query_biases.dimensions() != grad_query_biases.dimensions() ||
                    key_biases.dimensions() != grad_key_biases.dimensions() || 
                    val_biases.dimensions() != grad_val_biases.dimensions() )
                {
                    throw std::runtime_error("Bias and gradient dimension mismatch in multi-head attention layer: " + layer.get_layer_name());
                }
                layer.get_query_biases() -= learning_rate * layer.get_grad_query_biases();
                layer.get_key_biases() -= learning_rate * layer.get_grad_key_biases();
                layer.get_value_biases() -= learning_rate * layer.get_grad_value_biases();
            }
        }
    }
}