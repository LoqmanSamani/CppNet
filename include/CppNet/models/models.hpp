/**
 * @file models.hpp
 * @brief Sequential model implementation for CppNet
 *
 * Provides a container that chains layers and activations sequentially
 * for building, training, and evaluating neural networks.
 */

#ifndef MODELS_HPP
#define MODELS_HPP

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/layer.hpp"
#include "CppNet/activations/activation.hpp"
#include "CppNet/losses/loss.hpp"
#include "CppNet/optimizers/optimizer.hpp"

namespace CppNet
{
    namespace detail
    {
        /**
         * @class Step
         * @brief Type-erased interface for a sequential forward/backward step.
         */
        class Step
        {
        public:
            virtual Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input) = 0;
            virtual Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) = 0;
            virtual bool is_trainable() const = 0;
            virtual void step(Optimizers::Optimizer& optimizer, float learning_rate) = 0;
            virtual ~Step() = default;
        };

        /**
         * @class LayerStep
         * @brief Wraps a concrete layer type, preserving its forward/backward signatures.
         */
        template<typename LayerT>
        class LayerStep : public Step
        {
        public:
            explicit LayerStep(std::shared_ptr<LayerT> layer) : layer_(std::move(layer)) {}

            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input) override
            {
                return layer_->forward(input);
            }

            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) override
            {
                return layer_->backward(grad_output);
            }

            bool is_trainable() const override { return layer_->is_trainable(); }
            void step(Optimizers::Optimizer& opt, float lr) override { layer_->step(opt, lr); }

        private:
            std::shared_ptr<LayerT> layer_;
        };

        /**
         * @class ActivationStep
         * @brief Wraps an Activation so it can participate in a sequential pipeline.
         */
        class ActivationStep : public Step
        {
        public:
            explicit ActivationStep(std::shared_ptr<Activations::Activation> act)
                : activation_(std::move(act)) {}

            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input) override
            {
                return activation_->forward(input);
            }

            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output) override
            {
                return activation_->backward(grad_output);
            }

            bool is_trainable() const override { return false; }
            void step(Optimizers::Optimizer&, float) override {}

        private:
            std::shared_ptr<Activations::Activation> activation_;
        };
    } // namespace detail

    namespace Models
    {
        /**
         * @class SequentialModel
         * @brief A linear stack of layers and activations that forms a neural network.
         *
         * Layers and activations are added in order and executed sequentially
         * during forward and backward passes.
         *
         * @example
         *   SequentialModel model;
         *   model.add_layer(std::make_shared<Layers::Linear>(2, 64, "fc1"));
         *   model.add_activation(std::make_shared<Activations::ReLU>("cpu-eigen"));
         *   model.add_layer(std::make_shared<Layers::Linear>(64, 3, "fc2"));
         *   auto output = model.forward(input);
         *   auto grad   = model.backward(loss_grad);
         *   model.update(optimizer, lr);
         */
        class SequentialModel
        {
        public:
            SequentialModel() = default;

            /**
             * @brief Add a layer to the model (for update/get_layer only).
             *
             * Use this when the layer does not participate in the
             * model's sequential forward()/backward() pipeline
             * (e.g. Embedding, Attention, or layers with non-2D tensors).
             * The layer will still be updated by update() and accessible
             * via get_layer().
             */
            void add_layer(std::shared_ptr<Layers::Layer> layer);

            /**
             * @brief Add a concrete layer to the sequential pipeline.
             *
             * The template captures the actual layer type so that its
             * forward() / backward() can be called through the type-erased
             * Step interface without requiring virtual methods on Layer.
             * The layer also participates in update() and get_layer().
             *
             * @note Only enabled for layer types that support
             *       forward(const Eigen::Tensor<float, 2>&).
             */
            template<typename LayerT,
                     typename = std::enable_if_t<std::is_same<
                         std::decay_t<decltype(std::declval<LayerT>().forward(
                             std::declval<const Eigen::Tensor<float, 2>&>()))>,
                         Eigen::Tensor<float, 2>>::value>>
            void add_layer(std::shared_ptr<LayerT> layer)
            {
                if (!layer)
                    throw std::invalid_argument("Cannot add null layer to model");
                steps_.push_back(std::make_shared<detail::LayerStep<LayerT>>(layer));
                layers_.push_back(layer);
            }

            /**
             * @brief Add an activation to the model.
             * @param activation Shared pointer to an Activation object
             */
            void add_activation(std::shared_ptr<Activations::Activation> activation);

            /**
             * @brief Forward pass through all layers and activations (2D)
             * @param input Input tensor [batch, features]
             * @return Output tensor
             */
            Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& input);

            /**
             * @brief Backward pass through all layers and activations in reverse (2D)
             * @param grad_output Gradient from the loss
             * @return Gradient with respect to input
             */
            Eigen::Tensor<float, 2> backward(const Eigen::Tensor<float, 2>& grad_output);

            /**
             * @brief Get number of trainable layers in the model
             * @return Trainable layer count
             */
            std::size_t num_layers() const { return layers_.size(); }

            /**
             * @brief Get total number of steps (layers + activations)
             * @return Step count
             */
            std::size_t num_steps() const { return steps_.size(); }

            /**
             * @brief Get a trainable layer by index
             * @param index Zero-based index among trainable layers
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
             * @brief Zero all accumulated gradients in every trainable layer
             *
             * Call this at the start of each training iteration, before
             * the forward + backward pass.
             */
            void zero_grad();

            /**
             * @brief Print a summary of the model architecture
             */
            void summary() const;

        private:
            std::vector<std::shared_ptr<detail::Step>> steps_;
            std::vector<std::shared_ptr<Layers::Layer>> layers_;
        };
    }
}

#endif // MODELS_HPP
