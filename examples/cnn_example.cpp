#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include <iostream>
#include <chrono>
#include <Eigen/Dense>

int main() {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    Eigen::Tensor<double, 4> input_(32, 3, 100, 100);
    input_.setRandom();
    
    std::tuple<int, int> kernel_size = std::make_tuple(7, 7);
    std::tuple<int, int> stride = std::make_tuple(1, 1);
    std::tuple<int, int, int, int> num_padding = std::make_tuple(2, 2, 2, 2);
    
    CppNet::Layers::Conv2d conv(3, 20, nullptr, kernel_size, stride, "valid", num_padding, "zero", "conv2222", true, true);
    
    // Time the forward pass
    auto forward_start = std::chrono::high_resolution_clock::now();
    Eigen::Tensor<double, 4> forw = conv.forward(input_);
    auto forward_end = std::chrono::high_resolution_clock::now();
    
    // Time the backward pass
    auto backward_start = std::chrono::high_resolution_clock::now();
    Eigen::Tensor<double, 4> back = conv.backward(forw);
    auto backward_end = std::chrono::high_resolution_clock::now();
    
    // End total timing
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate durations
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto forward_duration = std::chrono::duration_cast<std::chrono::microseconds>(forward_end - forward_start);
    auto backward_duration = std::chrono::duration_cast<std::chrono::microseconds>(backward_end - backward_start);
    
    std::cout << "Forward dimensions: " << forw.dimensions() << std::endl;
    std::cout << "Backward dimensions: " << back.dimensions() << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << " microseconds" << std::endl;
    std::cout << "Forward pass time: " << forward_duration.count() << " microseconds" << std::endl;
    std::cout << "Backward pass time: " << backward_duration.count() << " microseconds" << std::endl;
    
    return 0;
}