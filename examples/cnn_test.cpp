#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <vector>
#include <iomanip>
#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h>

#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <random>


Eigen::Tensor<double, 2> generate_one_hot_labels(int batch_size, int num_classes) {
    Eigen::Tensor<double, 2> labels(batch_size, num_classes);
    labels.setZero();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_classes - 1);
    
    for (int i = 0; i < batch_size; ++i) {
        int class_idx = dis(gen);
        labels(i, class_idx) = 1.0;
    }
    
    return labels;
}

// Function to calculate correct flattened size
int calculate_flattened_size(int input_h, int input_w) {
    // Conv1: 64x64 -> (64-5+1) = 60x60
    int after_conv1_h = input_h - 5 + 1;
    int after_conv1_w = input_w - 5 + 1;
    
    // Pool1: 60x60 -> 30x30 (stride 2)
    int after_pool1_h = after_conv1_h / 2;
    int after_pool1_w = after_conv1_w / 2;
    
    // Conv2: 30x30 -> (30-5+1) = 26x26  
    int after_conv2_h = after_pool1_h - 5 + 1;
    int after_conv2_w = after_pool1_w - 5 + 1;
    
    // Pool2: 26x26 -> 13x13 (stride 2)
    int after_pool2_h = after_conv2_h / 2;
    int after_pool2_w = after_conv2_w / 2;
    
    // Flattened: 32 channels * 13 * 13
    return 32 * after_pool2_h * after_pool2_w;
}

// Function to run training with specific number of threads (cleaner version)
double run_training_benchmark(int num_threads, int epochs_to_test = 10, bool verbose = false) {
    omp_set_num_threads(num_threads);
    
    const int batch_size = 16;
    const int num_classes = 10;
    const int flattened_size = calculate_flattened_size(64, 64);
    
    // Generate data
    Eigen::Tensor<double, 4> input(batch_size, 3, 64, 64);
    input.setRandom();
    Eigen::Tensor<double, 2> targets = generate_one_hot_labels(batch_size, num_classes);

    // Create network layers
    CppNet::Layers::Conv2d conv1(3, 16, std::make_tuple(5, 5), std::make_tuple(1, 1), "conv1");
    CppNet::Activations::ReLU relu1;
    CppNet::Layers::MaxPool2D pool1(std::make_tuple(2, 2), std::make_tuple(2, 2), "pool1");
    
    CppNet::Layers::Conv2d conv2(16, 32, std::make_tuple(5, 5), std::make_tuple(1, 1), "conv2");
    CppNet::Activations::ReLU relu2;
    CppNet::Layers::MaxPool2D pool2(std::make_tuple(2, 2), std::make_tuple(2, 2), "pool2");
    
    CppNet::Layers::Flatten flatten("flatten");
    CppNet::Layers::Linear fc1(flattened_size, 128, "fc1"); // Use calculated size
    CppNet::Activations::ReLU relu3;
    CppNet::Layers::Linear fc2(128, num_classes, "fc2");
    CppNet::Activations::Softmax softmax;
    
    CppNet::Losses::CategoricalCrossEntropy loss("mean", false, 0.0);

    // Set thread counts for all layers
    conv1.set_num_threads(num_threads);
    pool1.set_num_threads(num_threads);
    relu1.set_num_threads(num_threads);
    conv2.set_num_threads(num_threads);
    pool2.set_num_threads(num_threads);
    relu2.set_num_threads(num_threads);
    flatten.set_num_threads(num_threads);
    relu3.set_num_threads(num_threads);
    fc1.set_num_threads(num_threads);
    fc2.set_num_threads(num_threads);
    softmax.set_num_threads(num_threads);
    loss.set_num_threads(num_threads);

    // Warmup run (not timed)
    if (verbose) std::cout << "Warmup run..." << std::endl;
    auto warmup_out1 = conv1.forward(input);
    auto warmup_out2 = relu1.forward(warmup_out1);
    auto warmup_out3 = pool1.forward(warmup_out2);
    auto warmup_out4 = conv2.forward(warmup_out3);
    auto warmup_out5 = relu2.forward(warmup_out4);
    auto warmup_out6 = pool2.forward(warmup_out5);
    auto warmup_out7 = flatten.forward(warmup_out6);
    auto warmup_out8 = fc1.forward(warmup_out7);
    auto warmup_out9 = relu3.forward(warmup_out8);
    auto warmup_out10 = fc2.forward(warmup_out9);
    auto warmup_predictions = softmax.forward(warmup_out10);
    double warmup_loss = loss.forward(warmup_predictions, targets);
    if (verbose) std::cout << "Warmup loss: " << warmup_loss << std::endl;
    
    // Start actual timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double total_loss = 0.0;
    
    for (int epoch = 0; epoch < epochs_to_test; ++epoch) 
    {
        // Forward pass (no debug output during timing)
        auto conv1_out = conv1.forward(input);
        auto relu1_out = relu1.forward(conv1_out);
        auto pool1_out = pool1.forward(relu1_out);
        
        auto conv2_out = conv2.forward(pool1_out);
        auto relu2_out = relu2.forward(conv2_out);
        auto pool2_out = pool2.forward(relu2_out);
        
        auto flatten_out = flatten.forward(pool2_out);
        auto fc1_out = fc1.forward(flatten_out);
        auto relu3_out = relu3.forward(fc1_out);
        auto fc2_out = fc2.forward(relu3_out);
        auto predictions = softmax.forward(fc2_out);
        
        // Loss computation
        double batch_loss = loss.forward(predictions, targets);
        total_loss += batch_loss;
        
        // Backward pass
        auto loss_grad = loss.backward(predictions, targets);
        auto softmax_grad = softmax.backward(loss_grad);
        auto fc2_grad = fc2.backward(softmax_grad);
        auto relu3_grad = relu3.backward(fc2_grad);
        auto fc1_grad = fc1.backward(relu3_grad);
        auto flatten_grad = flatten.backward(fc1_grad);
        auto pool2_grad = pool2.backward(flatten_grad);
        auto relu2_grad = relu2.backward(pool2_grad);
        auto conv2_grad = conv2.backward(relu2_grad);
        auto pool1_grad = pool1.backward(conv2_grad);
        auto relu1_grad = relu1.backward(pool1_grad);
        auto conv1_grad = conv1.backward(relu1_grad);
        
        // Reset gradients
        conv1.reset_grads();
        conv2.reset_grads();
        fc1.reset_grads();
        fc2.reset_grads();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double time_seconds = duration.count() / 1000.0;
    
    if (verbose) {
        double avg_loss = total_loss / epochs_to_test;
        std::cout << "Threads: " << num_threads 
                  << ", Epochs: " << epochs_to_test
                  << ", Avg Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << ", Time: " << std::setprecision(2) << time_seconds << "s" << std::endl;
    }
    
    return time_seconds;
}

int main() {
    std::cout << "=== CNN Library Multi-threading Benchmark ===" << std::endl;
    std::cout << "Expected flattened size: " << calculate_flattened_size(64, 64) << std::endl;
    
    std::vector<int> thread_counts = {1, 2, 4, 6, 8};
    const int trials = 3;
    const int epochs = 15; 
    
    std::cout << "\nRunning " << trials << " trials with " << epochs << " epochs each..." << std::endl;
    
    // Results storage
    std::vector<std::vector<double>> all_times(thread_counts.size());
    std::vector<double> mean_times(thread_counts.size());
    std::vector<double> std_times(thread_counts.size());
    
    for (size_t i = 0; i < thread_counts.size(); ++i) {
        std::cout << "\nTesting " << thread_counts[i] << " threads..." << std::endl;
        
        for (int trial = 0; trial < trials; ++trial) {
            double time = run_training_benchmark(thread_counts[i], epochs, trial == 0);
            all_times[i].push_back(time);
        }
        
        // Calculate statistics
        double sum = 0.0;
        for (double t : all_times[i]) sum += t;
        mean_times[i] = sum / trials;
        
        double var_sum = 0.0;
        for (double t : all_times[i]) {
            var_sum += (t - mean_times[i]) * (t - mean_times[i]);
        }
        std_times[i] = std::sqrt(var_sum / trials);
    }
    
    // Print results
    std::cout << "\n=== BENCHMARK RESULTS ===" << std::endl;
    std::cout << std::setw(8) << "Threads" << std::setw(12) << "Mean Time" << std::setw(8) << "Std Dev"
              << std::setw(10) << "Speedup" << std::setw(12) << "Efficiency" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    double baseline_time = mean_times[0];
    
    for (size_t i = 0; i < thread_counts.size(); ++i) {
        double speedup = baseline_time / mean_times[i];
        double efficiency = speedup / thread_counts[i] * 100.0;
        
        std::cout << std::setw(8) << thread_counts[i] 
                  << std::setw(12) << std::fixed << std::setprecision(2) << mean_times[i]
                  << std::setw(8) << std::setprecision(2) << std_times[i]
                  << std::setw(10) << std::setprecision(2) << speedup
                  << std::setw(11) << std::setprecision(1) << efficiency << "%" << std::endl;
    }
    
    return 0;
}

/*
=== CNN Library Multi-threading Benchmark ===
Expected flattened size: 5408

Running 3 trials with 15 epochs each...

Testing 1 threads...
Warmup run...
Warmup loss: 2.28259
Threads: 1, Epochs: 15, Avg Loss: 2.2826, Time: 40.98s

Testing 2 threads...
Warmup run...
Warmup loss: 2.34
Threads: 2, Epochs: 15, Avg Loss: 2.3433, Time: 24.32s

Testing 4 threads...
Warmup run...
Warmup loss: 2.43
Threads: 4, Epochs: 15, Avg Loss: 2.4282, Time: 18.90s

Testing 6 threads...
Warmup run...
Warmup loss: 2.35
Threads: 6, Epochs: 15, Avg Loss: 2.3520, Time: 24.74s

Testing 8 threads...
Warmup run...
Warmup loss: 2.32
Threads: 8, Epochs: 15, Avg Loss: 2.3217, Time: 14.45s

=== BENCHMARK RESULTS ===
 Threads   Mean Time Std Dev   Speedup  Efficiency
-------------------------------------------------------
       1       42.87    1.68      1.00      100.0%
       2       24.79    1.32      1.73       86.5%
       4       18.80    0.29      2.28       57.0%
       6       24.81    2.24      1.73       28.8%
       8       14.95    0.72      2.87       35.8%
*/