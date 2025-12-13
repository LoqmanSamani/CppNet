#include "CppNet/optimizers/sgd.hpp"
#include "CppNet/kernels/gpu/gpu.hpp"




namespace CppNet
{
    namespace Optimizers
    {
    
        SGD::SGD(int gpu_block_size): gpu_block_size_(gpu_block_size){}
        
        void SGD::step(CppNet::Layers::Linear& layer, float learning_rate)
        {
            if (!layer.is_trainable())
            {
                return;
            }

            std::string device = layer.get_device();
            if (device == "cpu")
            {
                // validate dimensions
                const auto& weights = layer.get_weights();
                const auto& grad_weights = layer.get_grad_weights();
                //if (weights.dimensions() != grad_weights.dimensions())
                //{
                //    throw std::runtime_error("Weight and gradient dimension mismatch in Linear layer: " + layer.get_layer_name());
                //}

                layer.get_weights() -= learning_rate * layer.get_grad_weights();

                if (layer.has_bias())
                {
                    auto& b = layer.get_biases();
                    const auto& db = layer.get_grad_biases();
                    //if (b.dimensions() != db.dimensions())
                    //{
                    //    throw std::runtime_error("Bias and gradient dimension mismatch in Linear layer: " + layer.get_layer_name());
                    //}
                    b -= learning_rate * db;
                }
            }
            else
            {
                step_gpu(layer, learning_rate);
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

        /*

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

            // get references to tensors
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
        */
    }
}