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

        void SequentialModel::add_activation(std::shared_ptr<Activations::Activation> activation)
        {
            if (!activation)
            {
                throw std::invalid_argument("Cannot add null activation to model");
            }
            steps_.push_back(std::make_shared<detail::ActivationStep>(std::move(activation)));
        }

        Eigen::Tensor<float, 2> SequentialModel::forward(const Eigen::Tensor<float, 2>& input)
        {
            Eigen::Tensor<float, 2> out = input;
            for (auto& step : steps_)
            {
                out = step->forward(out);
            }
            return out;
        }

        Eigen::Tensor<float, 2> SequentialModel::backward(const Eigen::Tensor<float, 2>& grad_output)
        {
            Eigen::Tensor<float, 2> grad = grad_output;
            for (auto it = steps_.rbegin(); it != steps_.rend(); ++it)
            {
                grad = (*it)->backward(grad);
            }
            return grad;
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

        void SequentialModel::zero_grad()
        {
            for (auto& layer : layers_)
            {
                layer->reset_grads();
            }
        }

        void SequentialModel::summary() const
        {
            std::cout << "===============================================" << std::endl;
            std::cout << "SequentialModel Summary" << std::endl;
            std::cout << "===============================================" << std::endl;
            std::cout << "Total steps: " << steps_.size()
                      << " (" << layers_.size() << " layers)" << std::endl;

            int trainable_count = 0;
            int step_idx = 0;
            for (const auto& s : steps_)
            {
                std::cout << "  Step " << step_idx++ << ": "
                          << (s->is_trainable() ? "trainable layer" :
                              (!s->is_trainable() ? "activation/frozen" : ""))
                          << std::endl;
                if (s->is_trainable())
                    ++trainable_count;
            }
            std::cout << "Trainable layers: " << trainable_count << std::endl;
            std::cout << "===============================================" << std::endl;
        }
    }
}
