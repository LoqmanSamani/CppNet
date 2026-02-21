/**
 * @file init.cpp
 * @brief Implementation of weight initialization strategies
 */

#include "CppNet/utils/init.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

namespace CppNet
{
    namespace Utils
    {
        static std::mt19937& get_rng()
        {
            static std::mt19937 rng(std::random_device{}());
            return rng;
        }

        Eigen::Tensor<float, 2> xavier_uniform(int rows, int cols)
        {
            // Glorot uniform: U[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
            float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
            std::uniform_real_distribution<float> dist(-limit, limit);

            Eigen::Tensor<float, 2> weights(rows, cols);
            auto& rng = get_rng();
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    weights(i, j) = dist(rng);

            return weights;
        }

        Eigen::Tensor<float, 2> xavier_normal(int rows, int cols)
        {
            // Glorot normal: N(0, sqrt(2 / (fan_in + fan_out)))
            float stddev = std::sqrt(2.0f / static_cast<float>(rows + cols));
            std::normal_distribution<float> dist(0.0f, stddev);

            Eigen::Tensor<float, 2> weights(rows, cols);
            auto& rng = get_rng();
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    weights(i, j) = dist(rng);

            return weights;
        }

        Eigen::Tensor<float, 2> he_uniform(int rows, int cols)
        {
            // Kaiming uniform: U[-limit, limit] where limit = sqrt(6 / fan_in)
            float limit = std::sqrt(6.0f / static_cast<float>(rows));
            std::uniform_real_distribution<float> dist(-limit, limit);

            Eigen::Tensor<float, 2> weights(rows, cols);
            auto& rng = get_rng();
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    weights(i, j) = dist(rng);

            return weights;
        }

        Eigen::Tensor<float, 2> he_normal(int rows, int cols)
        {
            // Kaiming normal: N(0, sqrt(2 / fan_in))
            float stddev = std::sqrt(2.0f / static_cast<float>(rows));
            std::normal_distribution<float> dist(0.0f, stddev);

            Eigen::Tensor<float, 2> weights(rows, cols);
            auto& rng = get_rng();
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    weights(i, j) = dist(rng);

            return weights;
        }

        Eigen::Tensor<float, 2> uniform_init(int rows, int cols, float low, float high)
        {
            std::uniform_real_distribution<float> dist(low, high);

            Eigen::Tensor<float, 2> weights(rows, cols);
            auto& rng = get_rng();
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    weights(i, j) = dist(rng);

            return weights;
        }

        Eigen::Tensor<float, 2> normal_init(int rows, int cols, float mean, float stddev)
        {
            std::normal_distribution<float> dist(mean, stddev);

            Eigen::Tensor<float, 2> weights(rows, cols);
            auto& rng = get_rng();
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    weights(i, j) = dist(rng);

            return weights;
        }

        Eigen::Tensor<float, 2> constant_init(int rows, int cols, float value)
        {
            Eigen::Tensor<float, 2> weights(rows, cols);
            weights.setConstant(value);
            return weights;
        }

        Eigen::Tensor<float, 2> init_weights(int rows, int cols, const std::string& method)
        {
            if (method == "xavier" || method == "glorot")
                return xavier_uniform(rows, cols);
            else if (method == "xavier_normal" || method == "glorot_normal")
                return xavier_normal(rows, cols);
            else if (method == "he" || method == "kaiming")
                return he_uniform(rows, cols);
            else if (method == "he_normal" || method == "kaiming_normal")
                return he_normal(rows, cols);
            else if (method == "uniform")
                return uniform_init(rows, cols);
            else if (method == "normal")
                return normal_init(rows, cols);
            else if (method == "zeros")
                return constant_init(rows, cols, 0.0f);
            else if (method == "ones")
                return constant_init(rows, cols, 1.0f);
            else
                throw std::invalid_argument("Unknown weight initialization method: " + method);
        }
    }
}
