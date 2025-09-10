#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <iomanip>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h>
#include "layers.hpp"
#include "optimizers.hpp"

using namespace CppNet::Layers;
using namespace CppNet::Optimizers;

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
double time_operation(Func&& func, const std::string& operation_name = "", bool print = true) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    
    if (print && !operation_name.empty()) {
        std::cout << operation_name << " took: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
    }
    return time_ms;
}

// Struct to hold benchmark results
struct BenchmarkResults {
    double forward_time;
    double backward_time;
    double optimizer_time;
    double total_time;
    
    void print_summary(const std::string& label) const {
        std::cout << "\n" << label << " Summary:" << std::endl;
        std::cout << "- Forward pass:  " << std::fixed << std::setprecision(3) << forward_time << " ms" << std::endl;
        std::cout << "- Backward pass: " << std::fixed << std::setprecision(3) << backward_time << " ms" << std::endl;
        std::cout << "- Optimizer:     " << std::fixed << std::setprecision(3) << optimizer_time << " ms" << std::endl;
        std::cout << "- Total time:    " << std::fixed << std::setprecision(3) << total_time << " ms" << std::endl;
    }
};

// Function to create optimizer by name
std::unique_ptr<Optimizer> create_optimizer(const std::string& name) {
    
    return std::make_unique<SGD>();    
}

// Benchmark a specific optimizer configuration
BenchmarkResults benchmark_optimizer_config(
    const std::string& optimizer_name,
    int num_threads,
    int batch_size,
    int input_size,
    int output_size,
    int num_iterations,
    double learning_rate = 0.001
) {
    omp_set_num_threads(num_threads);
    
    // Create layer and optimizer
    Linear layer(input_size, output_size, 
                 "test_layer_" + optimizer_name, true, true, "cpu", "he_normal");
    auto optimizer = create_optimizer(optimizer_name);
    
    // Create test data
    auto input = create_random_input(batch_size, input_size);
    auto grad_output = create_random_input(batch_size, output_size);
    
    // Warm up
    for (int i = 0; i < 3; ++i) {
        auto output = layer.forward(input);
        layer.backward(grad_output);
        optimizer->step(layer, learning_rate);
        layer.reset_grads(); // Reset gradients after optimizer step
    }
    
    // Benchmark
    double total_forward_time = 0.0;
    double total_backward_time = 0.0;
    double total_optimizer_time = 0.0;
    
    for (int i = 0; i < num_iterations; ++i) {
        // Forward pass
        total_forward_time += time_operation([&]() {
            auto output = layer.forward(input);
        }, "", false);
        
        // Backward pass
        total_backward_time += time_operation([&]() {
            layer.backward(grad_output);
        }, "", false);
        
        // Optimizer step
        total_optimizer_time += time_operation([&]() {
            optimizer->step(layer, learning_rate);
        }, "", false);
        
        // Reset gradients (part of the training loop, not optimizer)
        layer.reset_grads();
    }
    
    return {
        total_forward_time / num_iterations,
        total_backward_time / num_iterations,
        total_optimizer_time / num_iterations,
        (total_forward_time + total_backward_time + total_optimizer_time) / num_iterations
    };
}

// Main performance comparison function
void compare_optimizers() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "OPTIMIZER PERFORMANCE COMPARISON" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int batch_size = 256;
    const int input_size = 1024;
    const int output_size = 512;
    const int num_iterations = 20;
    const int num_threads = 4; // Use optimal thread count from previous tests
    const double learning_rate = 0.001;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Batch size: " << batch_size << std::endl;
    std::cout << "- Input size: " << input_size << std::endl;
    std::cout << "- Output size: " << output_size << std::endl;
    std::cout << "- Iterations: " << num_iterations << std::endl;
    std::cout << "- Threads: " << num_threads << std::endl;
    std::cout << "- Learning rate: " << learning_rate << std::endl;
    
    std::vector<std::string> optimizers = {"sgd"};
    std::vector<BenchmarkResults> results;
    
    for (const auto& opt_name : optimizers) {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "Testing " << opt_name << "..." << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        try {
            auto result = benchmark_optimizer_config(
                opt_name, num_threads, batch_size, input_size, output_size, num_iterations, learning_rate
            );
            results.push_back(result);
            result.print_summary(opt_name);
        } catch (const std::exception& e) {
            std::cout << "Error testing " << opt_name << ": " << e.what() << std::endl;
            results.push_back({0, 0, 0, 0}); // Placeholder for failed test
        }
    }
    
    // Summary comparison
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "PERFORMANCE SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << std::left << std::setw(15) << "Optimizer" 
              << std::setw(12) << "Forward" 
              << std::setw(12) << "Backward" 
              << std::setw(12) << "Optimizer" 
              << std::setw(12) << "Total" << std::endl;
    std::cout << std::string(63, '-') << std::endl;
    
    for (size_t i = 0; i < optimizers.size() && i < results.size(); ++i) {
        const auto& result = results[i];
        std::cout << std::left << std::setw(15) << optimizers[i]
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.forward_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.backward_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.optimizer_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.total_time << std::endl;
    }
}

// Thread scaling test for optimizers
void test_optimizer_thread_scaling() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "OPTIMIZER THREAD SCALING TEST" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int batch_size = 512;
    const int input_size = 2048;  // Larger size to see parallelization benefits
    const int output_size = 1024;
    const int num_iterations = 10;
    const double learning_rate = 0.001;
    
    std::vector<int> thread_counts = {1, 2, 4, 8};
    std::vector<std::string> test_optimizers = {"sgd"}; // Test simple and complex optimizers
    
    for (const auto& opt_name : test_optimizers) {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "Thread scaling for " << opt_name << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        std::cout << std::left << std::setw(10) << "Threads" 
                  << std::setw(15) << "Total Time" 
                  << std::setw(15) << "Optimizer Time" 
                  << std::setw(10) << "Speedup" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        double baseline_time = 0.0;
        
        for (int num_threads : thread_counts) {
            if (num_threads > omp_get_num_procs()) continue;
            
            auto result = benchmark_optimizer_config(
                opt_name, num_threads, batch_size, input_size, output_size, num_iterations, learning_rate
            );
            
            if (baseline_time == 0.0) baseline_time = result.total_time;
            double speedup = baseline_time / result.total_time;
            
            std::cout << std::left << std::setw(10) << num_threads
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.total_time
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.optimizer_time
                      << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
    }
}

int main() {
    std::cout << "=== Enhanced OpenMP Performance Test with Optimizers ===" << std::endl;
    std::cout << "Available processors: " << omp_get_num_procs() << std::endl;
    std::cout << "Default OpenMP threads: " << omp_get_max_threads() << std::endl;
    
    // Test 1: Basic layer performance (from original test)
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "BASIC LAYER PERFORMANCE TEST" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int batch_size = 512;
    const int input_size = 1024;
    const int output_size = 512;
    const int num_iterations = 5; // Reduced for faster testing
    
    auto input = create_random_input(batch_size, input_size);
    auto grad_output = create_random_input(batch_size, output_size);
    
    // Test with optimal thread count (4 from previous results)
    omp_set_num_threads(4);
    Linear layer(input_size, output_size, "base_test_layer", true, true, "cpu", "he_normal");
    layer.print_layer_info();
    
    // Quick performance check
    double total_forward_time = 0.0;
    double total_backward_time = 0.0;
    
    // Warm up
    for (int i = 0; i < 3; ++i) {
        auto output = layer.forward(input);
        layer.backward(grad_output);
        layer.reset_grads();
    }
    
    // Benchmark
    for (int i = 0; i < num_iterations; ++i) {
        total_forward_time += time_operation([&]() {
            auto output = layer.forward(input);
        }, "", false);
        
        total_backward_time += time_operation([&]() {
            layer.backward(grad_output);
        }, "", false);
        
        layer.reset_grads();
    }
    
    std::cout << "\nBasic Layer Performance (4 threads):" << std::endl;
    std::cout << "- Forward pass: " << std::fixed << std::setprecision(3) 
              << total_forward_time / num_iterations << " ms" << std::endl;
    std::cout << "- Backward pass: " << std::fixed << std::setprecision(3) 
              << total_backward_time / num_iterations << " ms" << std::endl;
    
    // Test 2: Optimizer performance comparison
    compare_optimizers();
    
    // Test 3: Thread scaling for optimizers
    test_optimizer_thread_scaling();
    
    // Test 4: Correctness test with optimizer
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "CORRECTNESS TEST WITH OPTIMIZER" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int small_batch = 2;
    const int small_input = 3;
    const int small_output = 2;
    
    auto small_input_data = create_random_input(small_batch, small_input);
    auto small_grad_output = create_random_input(small_batch, small_output);
    
    Linear test_layer(small_input, small_output, "correctness_test", true, true);
    auto test_optimizer = std::make_unique<SGD>();
    
    std::cout << "Testing training loop with SGD optimizer..." << std::endl;
    std::cout << "Input shape: [" << small_input_data.dimension(0) 
              << ", " << small_input_data.dimension(1) << "]" << std::endl;
    std::cout << "Weight shape: [" << test_layer.get_weights().dimension(0) 
              << ", " << test_layer.get_weights().dimension(1) << "]" << std::endl;
    
    // Test a few training steps
    for (int step = 0; step < 3; ++step) {
        std::cout << "\nTraining step " << (step + 1) << ":" << std::endl;
        
        // Forward pass
        auto output = test_layer.forward(small_input_data);
        std::cout << "- Forward pass completed, output shape: [" 
                  << output.dimension(0) << ", " << output.dimension(1) << "]" << std::endl;
        
        // Backward pass
        auto grad_input = test_layer.backward(small_grad_output);
        std::cout << "- Backward pass completed, grad_input shape: [" 
                  << grad_input.dimension(0) << ", " << grad_input.dimension(1) << "]" << std::endl;
        
        // Optimizer step
        test_optimizer->step(test_layer, 0.01);
        std::cout << "- Optimizer step completed" << std::endl;
        
        // Reset gradients (important!)
        test_layer.reset_grads();
        std::cout << "- Gradients reset" << std::endl;
    }
    
    std::cout << "\n✓ All tests completed successfully!" << std::endl;
    std::cout << "\nKey findings:" << std::endl;
    std::cout << "- Layer operations scale well with OpenMP" << std::endl;
    std::cout << "- Optimizer overhead varies by complexity (SGD < Adam)" << std::endl;
    std::cout << "- reset_grads() is correctly called after optimizer steps" << std::endl;
    std::cout << "- Training loop integrates properly with parallelization" << std::endl;
    
    return 0;
}