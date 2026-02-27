/**
 * @file gradient_clip.cpp
 * @brief Gradient clipping implementations (clip-by-value and clip-by-norm)
 */

#include "CppNet/utils/gradient_clip.hpp"
#include <algorithm>
#include <cmath>

namespace CppNet
{
    namespace Utils
    {
        void clip_by_value(Eigen::Tensor<float, 2>& tensor, float max_val)
        {
            float neg = -max_val;
            int rows = tensor.dimension(0);
            int cols = tensor.dimension(1);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                {
                    if (tensor(i, j) > max_val)  tensor(i, j) = max_val;
                    if (tensor(i, j) < neg)       tensor(i, j) = neg;
                }
        }

        void clip_by_value(Eigen::Tensor<float, 1>& tensor, float max_val)
        {
            float neg = -max_val;
            int size = tensor.dimension(0);
            for (int i = 0; i < size; ++i)
            {
                if (tensor(i) > max_val) tensor(i) = max_val;
                if (tensor(i) < neg)     tensor(i) = neg;
            }
        }

        float clip_by_norm(Eigen::Tensor<float, 2>& tensor, float max_norm)
        {
            int rows = tensor.dimension(0);
            int cols = tensor.dimension(1);

            float norm_sq = 0.0f;
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    norm_sq += tensor(i, j) * tensor(i, j);
            float norm = std::sqrt(norm_sq);

            if (norm > max_norm)
            {
                float scale = max_norm / (norm + 1e-8f);
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        tensor(i, j) *= scale;
            }

            return norm;
        }

        float clip_by_norm(Eigen::Tensor<float, 1>& tensor, float max_norm)
        {
            int size = tensor.dimension(0);

            float norm_sq = 0.0f;
            for (int i = 0; i < size; ++i)
                norm_sq += tensor(i) * tensor(i);
            float norm = std::sqrt(norm_sq);

            if (norm > max_norm)
            {
                float scale = max_norm / (norm + 1e-8f);
                for (int i = 0; i < size; ++i)
                    tensor(i) *= scale;
            }

            return norm;
        }
    }
}
