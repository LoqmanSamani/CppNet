/**
 * @file init.hpp
 * @brief Weight initialization strategies for CppNet
 *
 * Provides Xavier, He, uniform, normal, and constant weight
 * initialization functions.
 */

#ifndef INIT_HPP
#define INIT_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Utils
    {
        /**
         * @brief Xavier (Glorot) uniform initialization
         * @param rows Number of input features (fan_in)
         * @param cols Number of output features (fan_out)
         * @return Initialized weight tensor [rows, cols]
         */
        Eigen::Tensor<float, 2> xavier_uniform(int rows, int cols);

        /**
         * @brief Xavier (Glorot) normal initialization
         * @param rows Number of input features (fan_in)
         * @param cols Number of output features (fan_out)
         * @return Initialized weight tensor [rows, cols]
         */
        Eigen::Tensor<float, 2> xavier_normal(int rows, int cols);

        /**
         * @brief He (Kaiming) uniform initialization
         * @param rows Number of input features (fan_in)
         * @param cols Number of output features (fan_out)
         * @return Initialized weight tensor [rows, cols]
         */
        Eigen::Tensor<float, 2> he_uniform(int rows, int cols);

        /**
         * @brief He (Kaiming) normal initialization
         * @param rows Number of input features (fan_in)
         * @param cols Number of output features (fan_out)
         * @return Initialized weight tensor [rows, cols]
         */
        Eigen::Tensor<float, 2> he_normal(int rows, int cols);

        /**
         * @brief Uniform random initialization in [low, high)
         * @param rows Number of rows
         * @param cols Number of columns
         * @param low Lower bound (inclusive)
         * @param high Upper bound (exclusive)
         * @return Initialized weight tensor [rows, cols]
         */
        Eigen::Tensor<float, 2> uniform_init(int rows, int cols, float low = -1.0f, float high = 1.0f);

        /**
         * @brief Normal (Gaussian) random initialization
         * @param rows Number of rows
         * @param cols Number of columns
         * @param mean Mean of the distribution
         * @param stddev Standard deviation
         * @return Initialized weight tensor [rows, cols]
         */
        Eigen::Tensor<float, 2> normal_init(int rows, int cols, float mean = 0.0f, float stddev = 0.01f);

        /**
         * @brief Initialize all weights to a constant value
         * @param rows Number of rows
         * @param cols Number of columns
         * @param value Constant value
         * @return Initialized weight tensor [rows, cols]
         */
        Eigen::Tensor<float, 2> constant_init(int rows, int cols, float value = 0.0f);

        /**
         * @brief Initialize weights using a named strategy
         * @param rows Number of rows
         * @param cols Number of columns
         * @param method Initialization method name ("xavier", "he", "uniform", "normal", "zeros")
         * @return Initialized weight tensor [rows, cols]
         */
        Eigen::Tensor<float, 2> init_weights(int rows, int cols, const std::string& method = "xavier");
    }
}

#endif // INIT_HPP
