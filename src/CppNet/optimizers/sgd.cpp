#include "CppNet/optimizers/sgd.hpp"

namespace CppNet
{
    namespace Optimizers
    {

        SGD::SGD(int gpu_block_size): gpu_block_size_(gpu_block_size){}

        void SGD::update(float* weights, const float* gradients,
                         int size, float learning_rate)
        {
            for (int i = 0; i < size; ++i)
                weights[i] -= learning_rate * gradients[i];
        }

    }
}
