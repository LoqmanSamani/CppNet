/**
 * @file tensor_io.hpp
 * @brief Tensor serialization in NumPy .npy v1.0 format
 *
 * Allows saving and loading Eigen tensors as .npy files so they
 * can be exchanged with Python / NumPy.
 *
 * Supported: 1D and 2D float32 tensors  (dtype '<f4').
 *
 * Format spec: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
 */

#ifndef TENSOR_IO_HPP
#define TENSOR_IO_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace CppNet
{
    namespace Utils
    {
        // ---------------------------------------------------------------
        //  Save to .npy
        // ---------------------------------------------------------------

        /**
         * @brief Save a 1D tensor as a .npy file (float32, little-endian)
         * @param path File path (should end with .npy)
         * @param tensor [N]
         */
        void save_npy(const std::string& path, const Eigen::Tensor<float, 1>& tensor);

        /**
         * @brief Save a 2D tensor as a .npy file (float32, little-endian)
         * @param path File path (should end with .npy)
         * @param tensor [rows, cols]
         */
        void save_npy(const std::string& path, const Eigen::Tensor<float, 2>& tensor);

        /**
         * @brief Save a 3D tensor as a .npy file (float32, little-endian)
         * @param path File path (should end with .npy)
         * @param tensor [d0, d1, d2]
         */
        void save_npy(const std::string& path, const Eigen::Tensor<float, 3>& tensor);

        /**
         * @brief Save a 4D tensor as a .npy file (float32, little-endian)
         * @param path File path (should end with .npy)
         * @param tensor [d0, d1, d2, d3]
         */
        void save_npy(const std::string& path, const Eigen::Tensor<float, 4>& tensor);

        // ---------------------------------------------------------------
        //  Load from .npy
        // ---------------------------------------------------------------

        /** Load a 1D float32 tensor from a .npy file */
        Eigen::Tensor<float, 1> load_npy_1d(const std::string& path);

        /** Load a 2D float32 tensor from a .npy file */
        Eigen::Tensor<float, 2> load_npy_2d(const std::string& path);

        /** Load a 3D float32 tensor from a .npy file */
        Eigen::Tensor<float, 3> load_npy_3d(const std::string& path);

        /** Load a 4D float32 tensor from a .npy file */
        Eigen::Tensor<float, 4> load_npy_4d(const std::string& path);
    }
}

#endif // TENSOR_IO_HPP
