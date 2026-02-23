#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP


namespace CppNet
{
    namespace Optimizers
    {
        class Optimizer
        {
        public:
            /**
             * @brief Generic parameter update on raw arrays.
             * Allows all layer types to use any optimizer.
             * @param weights   Pointer to the parameter buffer
             * @param gradients Pointer to the gradient buffer (same size)
             * @param size      Number of scalar elements
             * @param learning_rate Step size
             */
            virtual void update(float* weights, const float* gradients,
                                int size, float learning_rate) = 0;

            virtual ~Optimizer() = default;
        };
    }
}

#endif // OPTIMIZER_HPP