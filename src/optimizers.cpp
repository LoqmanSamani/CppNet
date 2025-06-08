#include "optimizers.hpp"

namespace CppNet 
{
    namespace Optimizers
    {
        void SGD::update(Layers::Linear& layer, double learning_rate) 
        {
            if (layer.is_trainable()) 
            {
                layer.get_weights() -= learning_rate * layer.get_grad_weights();
                
                if (layer.has_bias()) 
                {
                    layer.get_biases() -= learning_rate * layer.get_grad_biases();
                }
            }
        }
    }   
}