/**
 * @file serialization.hpp
 * @brief Model save / load utilities for CppNet
 *
 * Saves and loads Eigen::Tensor weight data in a simple binary format:
 *   [4 bytes: ndims] [4 bytes each: dim0, dim1, ...] [float data...]
 *
 * Higher-level helpers save/load an entire SequentialModel by iterating
 * over its layers and writing/reading each weight tensor in order.
 */

#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// Forward declarations to avoid circular includes
namespace CppNet { namespace Models { class SequentialModel; } }

namespace CppNet
{
    namespace Utils
    {
        // ---------------------------------------------------------------
        //  Low-level tensor I/O
        // ---------------------------------------------------------------

        /** Save a 2D float tensor to a binary file */
        void save_tensor(const std::string& path, const Eigen::Tensor<float, 2>& tensor);

        /** Load a 2D float tensor from a binary file */
        Eigen::Tensor<float, 2> load_tensor_2d(const std::string& path);

        /** Save a 1D float tensor to a binary file */
        void save_tensor(const std::string& path, const Eigen::Tensor<float, 1>& tensor);

        /** Load a 1D float tensor from a binary file */
        Eigen::Tensor<float, 1> load_tensor_1d(const std::string& path);

        // ---------------------------------------------------------------
        //  Model-level I/O
        // ---------------------------------------------------------------

        /**
         * @brief Save all trainable weights of a SequentialModel
         *
         * Writes one binary file containing every weight tensor in order.
         * Format per tensor: [int32 ndims][int32 dims...][float data...]
         *
         * @param path  File path to write (e.g. "model.bin")
         * @param model The model whose weights to save
         */
        void save_model(const std::string& path, const Models::SequentialModel& model);

        /**
         * @brief Load weights into an existing SequentialModel
         *
         * The model must already have the same architecture (same number
         * and sizes of layers) as when save_model() was called.
         *
         * @param path  File path to read
         * @param model The model whose weights to overwrite
         */
        void load_model(const std::string& path, Models::SequentialModel& model);
    }
}

#endif // SERIALIZATION_HPP
