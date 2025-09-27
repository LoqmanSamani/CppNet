#include "optimizers.hpp"

namespace CppNet
{
    namespace Optimizers
    {
    
        //******************Stochastic Gradient Descent (SGD)*******************//
        void SGD::step(CppNet::Layers::Linear& layer, double learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }

            // Get references to tensors
            auto& weights = layer.get_weights();
            const auto& grad_weights = layer.get_grad_weights();
            if (weights.dimensions() != grad_weights.dimensions())
            {
                throw std::runtime_error("Weight and gradient dimension mismatch in Linear layer: " + layer.get_layer_name());
            }

            const int rows = weights.dimension(0);
            const int cols = weights.dimension(1);

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    weights(i, j) -= learning_rate * grad_weights(i, j);
                }
            }

            

            if (layer.has_bias())
            {
                auto& biases = layer.get_biases();
                const auto& grad_biases = layer.get_grad_biases();
                const int bias_size = biases.dimension(0);
                if (biases.dimensions() != grad_biases.dimensions())
                {
                    throw std::runtime_error("Bias and gradient dimension mismatch in Linear layer: " + layer.get_layer_name());
                }
        
                #pragma omp parallel for
                for (int i = 0; i < bias_size; ++i)
                {
                    biases(i) -= learning_rate * grad_biases(i);
                }
            }
        }

        void SGD::step(CppNet::Layers::Conv2d& layer, double learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }

            // validate dimensions
            auto& weights = layer.get_weights();
            const auto& grad_weights = layer.get_grad_weights();
            if (weights.dimensions() != grad_weights.dimensions())
            {
                throw std::runtime_error("Weight and gradient dimension mismatch in Conv2d layer: " + layer.get_layer_name());
            }

            const int rows = weights.dimension(0);
            const int cols = weights.dimension(1);
            const int depth = weights.dimension(2);
            const int num_filters = weights.dimension(3);

            #pragma omp parallel for collapse(4)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    for (int k = 0; k < depth; ++k)
                    {
                        for (int f = 0; f < num_filters; ++f)
                        {
                            weights(i, j, k, f) -= learning_rate * grad_weights(i, j, k, f);
                        }
                    }
                }
            }
            if (layer.has_bias())
            {
                auto& biases = layer.get_biases();
                const auto& grad_biases = layer.get_grad_biases();
                const int bias_size = biases.dimension(0);
                
                if (biases.dimensions() != grad_biases.dimensions())
                {
                    throw std::runtime_error("Bias and gradient dimension mismatch in Conv2d layer: " + layer.get_layer_name());
                }
        
                #pragma omp parallel for
                for (int i = 0; i < bias_size; ++i)
                {
                    biases(i) -= learning_rate * grad_biases(i);
                }
            }
        }

        void SGD::step(CppNet::Layers::MultiHeadAttention& layer, double learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }

            // Get references to tensors
            auto& query_weights = layer.get_query_weights();
            auto& key_weights = layer.get_key_weights();
            auto& val_weights = layer.get_value_weights();

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

            const int rows = query_weights.dimension(0);
            const int cols = query_weights.dimension(1);

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    query_weights(i, j) -= learning_rate * grad_query_weights(i, j);
                    key_weights(i, j) -= learning_rate * grad_key_weights(i, j);
                    val_weights(i, j) -= learning_rate * grad_val_weights(i, j);
                }
            }

            if (layer.has_bias())
            {
                auto& query_biases = layer.get_query_biases();
                auto& key_biases = layer.get_key_biases();
                auto& val_biases = layer.get_value_biases();

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
                
                const int bias_size = query_biases.dimension(0);

                #pragma omp parallel for
                for (int i = 0; i < bias_size; ++i)
                {
                    query_biases(i) -= learning_rate * grad_query_biases(i);
                    key_biases(i) -= learning_rate * grad_key_biases(i);
                    val_biases(i) -= learning_rate * grad_val_biases(i);
                }
            }
        }

        /************************************** Momentum *************************************/
        //Momentum::Momentum(double lr, double momentum) : lr(lr), momentum(momentum) {}

        /************************************** Adam *************************************/
        //Adam::Adam(double lr, double beta1, double beta2, double eps)
            //: lr(lr), beta1(beta1), beta2(beta2), eps(eps) {}

        
        /************************************** RMSProp *************************************/
        //RMSProp::RMSProp(double lr, double rho, double eps) : lr(lr), rho(rho), eps(eps) {}

    
        /************************************** Adagrad *************************************/
        //Adagrad::Adagrad(double lr, double eps) : lr(lr), eps(eps) {}

    }
}