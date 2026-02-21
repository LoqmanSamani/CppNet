/**
 * @file models.cpp
 * @brief Implementation of SequentialModel
 */

#include "CppNet/models/models.hpp"
#include <stdexcept>

namespace CppNet
{
    namespace Models
    {
        void SequentialModel::add_layer(std::shared_ptr<Layers::Layer> layer)
        {
            if (!layer)
            {
                throw std::invalid_argument("Cannot add null layer to model");
            }
            layers_.push_back(layer);
        }

        std::shared_ptr<Layers::Layer> SequentialModel::get_layer(std::size_t index) const
        {
            if (index >= layers_.size())
            {
                throw std::out_of_range("Layer index " + std::to_string(index) +
                                        " out of range (model has " + std::to_string(layers_.size()) + " layers)");
            }
            return layers_[index];
        }

        void SequentialModel::update(Optimizers::Optimizer& optimizer, float learning_rate)
        {
            for (auto& layer : layers_)
            {
                if (layer->is_trainable())
                {
                    layer->step(optimizer, learning_rate);
                }
            }
        }

        void SequentialModel::summary() const
        {
            std::cout << "===============================================" << std::endl;
            std::cout << "SequentialModel Summary" << std::endl;
            std::cout << "===============================================" << std::endl;
            std::cout << "Total layers: " << layers_.size() << std::endl;

            int trainable_count = 0;
            for (std::size_t i = 0; i < layers_.size(); ++i)
            {
                std::cout << "  Layer " << i << ": "
                          << (layers_[i]->is_trainable() ? "trainable" : "frozen")
                          << std::endl;
                if (layers_[i]->is_trainable())
                    ++trainable_count;
            }
            std::cout << "Trainable layers: " << trainable_count << std::endl;
            std::cout << "===============================================" << std::endl;
        }
    }
}
