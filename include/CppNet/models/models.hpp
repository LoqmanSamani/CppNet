/**
 * @file models.hpp
 * @brief Sequential model implementation for CppNet
 *
 * Provides a container that chains layers sequentially for
 * building, training, and evaluating neural networks.
 */

#ifndef MODELS_HPP
#define MODELS_HPP

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/layer.hpp"
#include "CppNet/losses/loss.hpp"
#include "CppNet/optimizers/optimizer.hpp"

namespace CppNet
{
    namespace Models
    {
        /**
         * @class SequentialModel
         * @brief A linear stack of layers that forms a neural network.
         *
         * Layers are added in order and executed sequentially during
         * forward and backward passes.
         *
         * @example
         *   SequentialModel model;
         *   model.add_layer(std::make_shared<Layers::Linear>(784, 128));
         *   auto output = model.forward(input);
         */
        class SequentialModel
        {
        public:
            SequentialModel() = default;

            /**
             * @brief Add a layer to the end of the model
             * @param layer Shared pointer to a Layer object
             */
            void add_layer(std::shared_ptr<Layers::Layer> layer);

            /**
             * @brief Get number of layers in the model
             * @return Layer count
             */
            std::size_t num_layers() const { return layers_.size(); }

            /**
             * @brief Get a layer by index
             * @param index Zero-based layer index
             * @return Shared pointer to the layer
             */
            std::shared_ptr<Layers::Layer> get_layer(std::size_t index) const;

            /**
             * @brief Update all trainable layers using the given optimizer
             * @param optimizer Reference to an Optimizer
             * @param learning_rate Learning rate for the update step
             */
            void update(Optimizers::Optimizer& optimizer, float learning_rate);

            /**
             * @brief Print a summary of the model architecture
             */
            void summary() const;

        private:
            std::vector<std::shared_ptr<Layers::Layer>> layers_;
        };
    }
}

#endif // MODELS_HPP
