/**
 * @file serialization.cpp
 * @brief Model save / load implementation
 *
 * Binary format per tensor:
 *   [int32_t ndims] [int32_t dim0] [int32_t dim1] ... [float * numel]
 *
 * Model file = concatenation of tensors from each trainable layer in order.
 * For Linear layers: weights (2D), then biases (1D).
 * For other trainable layers a generic approach writes what it can.
 */

#include "CppNet/utils/serialization.hpp"
#include "CppNet/models/models.hpp"
#include "CppNet/layers/linear.hpp"
#include "CppNet/layers/batch_norm.hpp"
#include "CppNet/layers/embedding.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace CppNet
{
    namespace Utils
    {
        void save_tensor(const std::string& path, const Eigen::Tensor<float, 2>& tensor)
        {
            std::ofstream ofs(path, std::ios::binary);
            if (!ofs) throw std::runtime_error("Cannot open file for writing: " + path);

            int32_t ndims = 2;
            int32_t d0 = static_cast<int32_t>(tensor.dimension(0));
            int32_t d1 = static_cast<int32_t>(tensor.dimension(1));
            ofs.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            ofs.write(reinterpret_cast<const char*>(&d0), sizeof(d0));
            ofs.write(reinterpret_cast<const char*>(&d1), sizeof(d1));
            ofs.write(reinterpret_cast<const char*>(tensor.data()),
                      static_cast<std::streamsize>(d0 * d1 * sizeof(float)));
        }

        Eigen::Tensor<float, 2> load_tensor_2d(const std::string& path)
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Cannot open file for reading: " + path);

            int32_t ndims;
            ifs.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            if (ndims != 2)
                throw std::runtime_error("Expected 2D tensor, got " + std::to_string(ndims) + "D");

            int32_t d0, d1;
            ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
            ifs.read(reinterpret_cast<char*>(&d1), sizeof(d1));

            Eigen::Tensor<float, 2> tensor(d0, d1);
            ifs.read(reinterpret_cast<char*>(tensor.data()),
                     static_cast<std::streamsize>(d0 * d1 * sizeof(float)));
            return tensor;
        }

        void save_tensor(const std::string& path, const Eigen::Tensor<float, 1>& tensor)
        {
            std::ofstream ofs(path, std::ios::binary);
            if (!ofs) throw std::runtime_error("Cannot open file for writing: " + path);

            int32_t ndims = 1;
            int32_t d0 = static_cast<int32_t>(tensor.dimension(0));
            ofs.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            ofs.write(reinterpret_cast<const char*>(&d0), sizeof(d0));
            ofs.write(reinterpret_cast<const char*>(tensor.data()),
                      static_cast<std::streamsize>(d0 * sizeof(float)));
        }

        Eigen::Tensor<float, 1> load_tensor_1d(const std::string& path)
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Cannot open file for reading: " + path);

            int32_t ndims;
            ifs.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            if (ndims != 1)
                throw std::runtime_error("Expected 1D tensor, got " + std::to_string(ndims) + "D");

            int32_t d0;
            ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));

            Eigen::Tensor<float, 1> tensor(d0);
            ifs.read(reinterpret_cast<char*>(tensor.data()),
                     static_cast<std::streamsize>(d0 * sizeof(float)));
            return tensor;
        }

        static void write_tensor_2d(std::ofstream& ofs, const Eigen::Tensor<float, 2>& t)
        {
            int32_t ndims = 2;
            int32_t d0 = static_cast<int32_t>(t.dimension(0));
            int32_t d1 = static_cast<int32_t>(t.dimension(1));
            ofs.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            ofs.write(reinterpret_cast<const char*>(&d0), sizeof(d0));
            ofs.write(reinterpret_cast<const char*>(&d1), sizeof(d1));
            ofs.write(reinterpret_cast<const char*>(t.data()),
                      static_cast<std::streamsize>(d0 * d1 * sizeof(float)));
        }

        static void write_tensor_1d(std::ofstream& ofs, const Eigen::Tensor<float, 1>& t)
        {
            int32_t ndims = 1;
            int32_t d0 = static_cast<int32_t>(t.dimension(0));
            ofs.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            ofs.write(reinterpret_cast<const char*>(&d0), sizeof(d0));
            ofs.write(reinterpret_cast<const char*>(t.data()),
                      static_cast<std::streamsize>(d0 * sizeof(float)));
        }

        static Eigen::Tensor<float, 2> read_tensor_2d(std::ifstream& ifs)
        {
            int32_t ndims;
            ifs.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            if (ndims != 2)
                throw std::runtime_error("Expected 2D tensor in model file");
            int32_t d0, d1;
            ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
            ifs.read(reinterpret_cast<char*>(&d1), sizeof(d1));
            Eigen::Tensor<float, 2> t(d0, d1);
            ifs.read(reinterpret_cast<char*>(t.data()),
                     static_cast<std::streamsize>(d0 * d1 * sizeof(float)));
            return t;
        }

        static Eigen::Tensor<float, 1> read_tensor_1d(std::ifstream& ifs)
        {
            int32_t ndims;
            ifs.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            if (ndims != 1)
                throw std::runtime_error("Expected 1D tensor in model file");
            int32_t d0;
            ifs.read(reinterpret_cast<char*>(&d0), sizeof(d0));
            Eigen::Tensor<float, 1> t(d0);
            ifs.read(reinterpret_cast<char*>(t.data()),
                     static_cast<std::streamsize>(d0 * sizeof(float)));
            return t;
        }

        void save_model(const std::string& path, const Models::SequentialModel& model)
        {
            std::ofstream ofs(path, std::ios::binary);
            if (!ofs) throw std::runtime_error("Cannot open file for writing: " + path);

            int32_t magic = 0x434E4554; // "CNET"
            int32_t nlayers = static_cast<int32_t>(model.num_layers());
            ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            ofs.write(reinterpret_cast<const char*>(&nlayers), sizeof(nlayers));

            for (std::size_t i = 0; i < model.num_layers(); ++i)
            {
                auto layer = model.get_layer(i);
                if (!layer->is_trainable()) continue;

                auto* linear = dynamic_cast<Layers::Linear*>(layer.get());
                if (linear)
                {
                    int32_t type_id = 1;
                    ofs.write(reinterpret_cast<const char*>(&type_id), sizeof(type_id));
                    write_tensor_2d(ofs, linear->get_weights());
                    if (linear->has_bias())
                        write_tensor_1d(ofs, linear->get_biases());
                    continue;
                }

                auto* bn = dynamic_cast<Layers::BatchNorm*>(layer.get());
                if (bn)
                {
                    int32_t type_id = 2; 
                    ofs.write(reinterpret_cast<const char*>(&type_id), sizeof(type_id));
                    write_tensor_1d(ofs, bn->get_gamma());
                    write_tensor_1d(ofs, bn->get_beta());
                    write_tensor_1d(ofs, bn->get_running_mean());
                    write_tensor_1d(ofs, bn->get_running_var());
                    continue;
                }

                auto* emb = dynamic_cast<Layers::Embedding*>(layer.get());
                if (emb)
                {
                    int32_t type_id = 3; 
                    ofs.write(reinterpret_cast<const char*>(&type_id), sizeof(type_id));
                    write_tensor_2d(ofs, emb->get_weight());
                    continue;
                }

                int32_t type_id = 0;
                ofs.write(reinterpret_cast<const char*>(&type_id), sizeof(type_id));
            }
        }

        void load_model(const std::string& path, Models::SequentialModel& model)
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Cannot open file for reading: " + path);

            int32_t magic, nlayers;
            ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            if (magic != 0x434E4554)
                throw std::runtime_error("Invalid model file (bad magic number)");
            ifs.read(reinterpret_cast<char*>(&nlayers), sizeof(nlayers));

            std::size_t layer_idx = 0;
            for (std::size_t i = 0; i < static_cast<std::size_t>(nlayers); ++i)
            {
                while (layer_idx < model.num_layers() &&
                       !model.get_layer(layer_idx)->is_trainable())
                    ++layer_idx;

                if (layer_idx >= model.num_layers()) break;

                auto layer = model.get_layer(layer_idx);

                int32_t type_id;
                ifs.read(reinterpret_cast<char*>(&type_id), sizeof(type_id));

                if (type_id == 1) 
                {
                    auto* linear = dynamic_cast<Layers::Linear*>(layer.get());
                    if (!linear)
                        throw std::runtime_error("Layer type mismatch at index " +
                                                 std::to_string(layer_idx));
                    linear->set_weights(read_tensor_2d(ifs));
                    if (linear->has_bias())
                        linear->set_biases(read_tensor_1d(ifs));
                }
                else if (type_id == 2)
                {
                    auto* bn = dynamic_cast<Layers::BatchNorm*>(layer.get());
                    if (!bn)
                        throw std::runtime_error("Layer type mismatch at index " +
                                                 std::to_string(layer_idx));
                    bn->get_gamma() = read_tensor_1d(ifs);
                    bn->get_beta() = read_tensor_1d(ifs);
                    auto running_mean = read_tensor_1d(ifs);
                    auto running_var = read_tensor_1d(ifs);
                    bn->set_running_mean(running_mean);
                    bn->set_running_var(running_var);
                }
                else if (type_id == 3)
                {
                    auto* emb = dynamic_cast<Layers::Embedding*>(layer.get());
                    if (!emb)
                        throw std::runtime_error("Layer type mismatch at index " +
                                                 std::to_string(layer_idx));
                    emb->set_weight(read_tensor_2d(ifs));
                }
                else if (type_id == 0) {}

                ++layer_idx;
            }
        }
    }
}
