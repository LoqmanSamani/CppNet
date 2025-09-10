#include "optimizers.hpp"

namespace CppNet
{
    namespace Optimizers
    {
    
        /************************************** SGD *************************************/
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