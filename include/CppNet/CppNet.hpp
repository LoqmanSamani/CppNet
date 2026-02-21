/**
 * @file CppNet.hpp
 * @brief Main header file for CppNet Deep Learning Library
 * 
 * This is the single include file for the entire CppNet library.
 * Include this file to get access to all CppNet functionality.
 * 
 * @example
 * #include <CppNet/CppNet.hpp>
 * 
 * int main() {
 *     // Create a neural network
 *     CppNet::Models::SequentialModel model;
 *     model.add_layer(std::make_shared<CppNet::Layers::Linear>(784, 128));
 *     // ... train your model
 * }
 */

#ifndef CPPNET_HPP
#define CPPNET_HPP

// Version information
#define CPPNET_VERSION_MAJOR 0
#define CPPNET_VERSION_MINOR 1
#define CPPNET_VERSION_PATCH 0

// Core dependencies
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>
#include <string>

// Activations
#include "CppNet/activations/activation.hpp"
#include "CppNet/activations/relu.hpp"
#include "CppNet/activations/sigmoid.hpp"
#include "CppNet/activations/softmax.hpp"
#include "CppNet/activations/tanh.hpp"
#include "CppNet/activations/leacky_relu.hpp"

// Layers
#include "CppNet/layers/layer.hpp"
#include "CppNet/layers/linear.hpp"
#include "CppNet/layers/conv2d.hpp"
#include "CppNet/layers/flatten.hpp"
#include "CppNet/layers/max_pool2d.hpp"
#include "CppNet/layers/rnn.hpp"
#include "CppNet/layers/attention.hpp"

// Losses
#include "CppNet/losses/loss.hpp"
#include "CppNet/losses/binary_cross_entropy.hpp"
#include "CppNet/losses/categorical_cross_entropy.hpp"
#include "CppNet/losses/mse.hpp"
#include "CppNet/losses/mae.hpp"
#include "CppNet/losses/huber.hpp"

// Optimizers
#include "CppNet/optimizers/optimizer.hpp"
#include "CppNet/optimizers/sgd.hpp"
#include "CppNet/optimizers/adam.hpp"
#include "CppNet/optimizers/adagrad.hpp"
#include "CppNet/optimizers/momentum.hpp"
#include "CppNet/optimizers/mrs_prop.hpp"

// Models
#include "CppNet/models/models.hpp"

// Metrics
#include "CppNet/metrics/metrics.hpp"

// Regularizations
#include "CppNet/regularizations/regularizations.hpp"

// Utils
#include "CppNet/utils/utils.hpp"
#include "CppNet/utils/init.hpp"
#include "CppNet/utils/elapsed_time.hpp"

// Visualizations
#include "CppNet/visualizations/visualizations.hpp"

// GPU Kernels (only if CUDA is enabled)
#ifdef USE_CUDA
#include "CppNet/kernels/gpu/gpu.hpp"
#endif

/**
 * @namespace CppNet
 * @brief Main namespace for the CppNet library
 */
namespace CppNet {

/**
 * @brief Get the version string of CppNet
 * @return Version string in format "major.minor.patch"
 */
inline std::string version() {
    return std::to_string(CPPNET_VERSION_MAJOR) + "." +
           std::to_string(CPPNET_VERSION_MINOR) + "." +
           std::to_string(CPPNET_VERSION_PATCH);
}

/**
 * @brief Check if CUDA support is available
 * @return true if compiled with CUDA support, false otherwise
 */
inline bool has_cuda_support() {
#ifdef USE_CUDA
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if OpenMP support is available
 * @return true if compiled with OpenMP support, false otherwise
 */
inline bool has_openmp_support() {
#ifdef USE_OPENMP
    return true;
#else
    return false;
#endif
}

} // namespace CppNet

#endif // CPPNET_HPP