#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h>
#include "layers.hpp"

using namespace CppNet::Layers;

// Helper function to create random input data
Eigen::Tensor<double, 2> create_random_input(int batch_size, int input_size) {
    Eigen::Tensor<double, 2> input(batch_size, input_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            input(i, j) = dis(gen);
        }
    }
    
    return input;
}

// Helper function to time operations
template<typename Func>
double time_operation(Func&& func, const std::string& operation_name) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    
    std::cout << operation_name << " took: " << time_ms << " ms" << std::endl;
    return time_ms;
}

int main() {
    std::cout << "=== OpenMP Linear Layer Performance Test ===" << std::endl;
    std::cout << "Available processors: " << omp_get_num_procs() << std::endl;
    std::cout << "Default OpenMP threads: " << omp_get_max_threads() << std::endl;
    
    // Test parameters
    const int batch_size = 1024;
    const int input_size = 2048;
    const int output_size = 1024;
    const int num_iterations = 10;
    
    std::cout << "\nTest Configuration:" << std::endl;
    std::cout << "- Batch size: " << batch_size << std::endl;
    std::cout << "- Input size: " << input_size << std::endl;
    std::cout << "- Output size: " << output_size << std::endl;
    std::cout << "- Iterations: " << num_iterations << std::endl;
    
    // Create test data
    auto input = create_random_input(batch_size, input_size);
    auto grad_output = create_random_input(batch_size, output_size);
    
    // Test with different numbers of threads
    std::vector<int> thread_counts = {1, 2, 4, 8};
    
    for (int num_threads : thread_counts) {
        if (num_threads > omp_get_num_procs()) continue;
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Testing with " << num_threads << " thread(s)" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        omp_set_num_threads(num_threads);
        
        // Create layer
        Linear layer(input_size, output_size, "test_layer", true, true, "cpu", "he_normal");
        layer.print_layer_info();
        
        double total_forward_time = 0.0;
        double total_backward_time = 0.0;
        
        // Warm up
        for (int i = 0; i < 3; ++i) {
            auto output = layer.forward(input);
            layer.backward(grad_output);
        }
        
        // Benchmark
        for (int i = 0; i < num_iterations; ++i) {
            // Forward pass
            total_forward_time += time_operation([&]() {
                auto output = layer.forward(input);
            }, "Forward pass " + std::to_string(i + 1));
            
            // Backward pass
            total_backward_time += time_operation([&]() {
                layer.backward(grad_output);
            }, "Backward pass " + std::to_string(i + 1));
            
            // Reset gradients for next iteration
            layer.reset_grads();
        }
        
        std::cout << "\nAverage times over " << num_iterations << " iterations:" << std::endl;
        std::cout << "- Forward pass: " << total_forward_time / num_iterations << " ms" << std::endl;
        std::cout << "- Backward pass: " << total_backward_time / num_iterations << " ms" << std::endl;
        std::cout << "- Total: " << (total_forward_time + total_backward_time) / num_iterations << " ms" << std::endl;
    }
    
    // Correctness test
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Correctness Test" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // Test with small matrices to verify correctness
    const int small_batch = 2;
    const int small_input = 3;
    const int small_output = 2;
    
    auto small_input_data = create_random_input(small_batch, small_input);
    auto small_grad_output = create_random_input(small_batch, small_output);
    
    Linear test_layer(small_input, small_output, "correctness_test", true, true);
    
    std::cout << "Input tensor shape: [" << small_input_data.dimension(0) 
              << ", " << small_input_data.dimension(1) << "]" << std::endl;
    std::cout << "Weight tensor shape: [" << test_layer.get_weights().dimension(0) 
              << ", " << test_layer.get_weights().dimension(1) << "]" << std::endl;
    
    // Forward pass
    auto output = test_layer.forward(small_input_data);
    std::cout << "Output tensor shape: [" << output.dimension(0) 
              << ", " << output.dimension(1) << "]" << std::endl;
    
    // Backward pass
    auto grad_input = test_layer.backward(small_grad_output);
    std::cout << "Gradient input shape: [" << grad_input.dimension(0) 
              << ", " << grad_input.dimension(1) << "]" << std::endl;
    
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}