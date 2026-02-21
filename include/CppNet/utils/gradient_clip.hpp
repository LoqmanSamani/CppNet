/**
 * @file gradient_clip.hpp
 * @brief Gradient clipping utilities for CppNet
 *
 * - clip_by_value:  Clamp each element to [-max_val, max_val]
 * - clip_by_norm:   Scale the entire tensor so its L2 norm ≤ max_norm
 */

#ifndef GRADIENT_CLIP_HPP
#define GRADIENT_CLIP_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>

namespace CppNet
{
    namespace Utils
    {
        /**
         * @brief Clip every element of a 2D tensor to [-max_val, max_val]
         * @param tensor  Gradient tensor (modified in-place)
         * @param max_val Maximum absolute value
         */
        void clip_by_value(Eigen::Tensor<float, 2>& tensor, float max_val);

        /**
         * @brief Clip every element of a 1D tensor to [-max_val, max_val]
         */
        void clip_by_value(Eigen::Tensor<float, 1>& tensor, float max_val);

        /**
         * @brief Scale a 2D tensor so its L2 norm does not exceed max_norm
         * @param tensor   Gradient tensor (modified in-place)
         * @param max_norm Maximum allowed L2 norm
         * @return The original norm before clipping
         */
        float clip_by_norm(Eigen::Tensor<float, 2>& tensor, float max_norm);

        /**
         * @brief Scale a 1D tensor so its L2 norm does not exceed max_norm
         */
        float clip_by_norm(Eigen::Tensor<float, 1>& tensor, float max_norm);
    }
}

#endif // GRADIENT_CLIP_HPP
