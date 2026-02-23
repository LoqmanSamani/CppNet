#include "CppNet/optimizers/sgd.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"




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
        
        void SGD::step(CppNet::Layers::Linear& layer, float learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }

            std::string device = layer.get_device();
            if (device == "cpu" || device == "cpu-eigen")
            {
                // validate dimensions
                const auto& weights = layer.get_weights();
                const auto& grad_weights = layer.get_grad_weights();
    

                layer.get_weights() -= learning_rate * layer.get_grad_weights();

                if (layer.has_bias())
                {
                    auto& b = layer.get_biases();
                    const auto& db = layer.get_grad_biases();
                    
                    b -= learning_rate * db;
                }
            }
            else
            {
                #ifdef USE_CUDA
                step_gpu(layer, learning_rate);
                #else
                throw std::runtime_error("GPU not available. Rebuild with CUDA support or use device=\"cpu\" / \"cpu-eigen\".");
                #endif
            }
        }

        #ifdef USE_CUDA
        void SGD::step_gpu(CppNet::Layers::Linear& layer, float learning_rate)
        {
            if (!layer.is_gpu_initialized())
            {
                throw std::runtime_error("GPU buffers not initialized for layer: " + layer.get_layer_name());
            }

            float* w = layer.get_d_weights();
            float* dw = layer.get_d_grad_weights();

            const int rows = layer.get_weights().dimension(0);
            const int cols = layer.get_weights().dimension(1);

            const int total_params = rows * cols;

            const int grid_size = (total_params + gpu_block_size_ - 1) / gpu_block_size_;

            CppNet::Kernels::GPU::sgd_step_kernel<<<grid_size, gpu_block_size_>>>(w, dw, learning_rate, total_params);

            if (layer.has_bias())
            {
                float* b = layer.get_d_bias();
                float* db = layer.get_d_grad_bias();

                const int bsize = layer.get_biases().dimension(0);
                const int bgrid_size = (bsize + gpu_block_size_ - 1) / gpu_block_size_;

                CppNet::Kernels::GPU::sgd_step_kernel<<<bgrid_size, gpu_block_size_>>>(b, db, learning_rate, bsize);
            }

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
            }
        }
        #endif

    }
}