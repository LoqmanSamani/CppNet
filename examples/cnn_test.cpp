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




/*

// Function to generate random one-hot encoded labels
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

// Function to run training with specific number of threads
double run_training_with_threads(int num_threads, int epochs_to_test = 5) {
    std::cout << "\n=== Testing with " << num_threads << " threads ===" << std::endl;
    
    // Set OpenMP thread count
    omp_set_num_threads(num_threads);
    
    // Generate input data: batch_size=16, channels=3, height=64, width=64
    const int batch_size = 16;
    const int num_classes = 10;
    Eigen::Tensor<double, 4> input(batch_size, 3, 64, 64);
    input.setRandom();
    
    // Generate target labels (one-hot encoded)
    Eigen::Tensor<double, 2> targets = generate_one_hot_labels(batch_size, num_classes);

    // Define simplified network architecture
    // Conv1 -> ReLU -> MaxPool -> Conv2 -> ReLU -> MaxPool -> Flatten -> FC1 -> ReLU -> FC2 -> Softmax
    
    CppNet::Layers::Conv2d conv1(
        3, 16, std::make_tuple(5, 5), std::make_tuple(1, 1),
        "conv1", true, true, "valid", std::make_tuple(2, 2, 2, 2),
        "zero", "cpu", "xavier"
    );
    CppNet::Activations::ReLU relu1;
   
    CppNet::Layers::MaxPool2D pool1(
        std::make_tuple(2, 2), std::make_tuple(2, 2),
        "pool1", "valid", std::make_tuple(0, 0, 0, 0),
        "zero", "cpu"
    );

    CppNet::Layers::Conv2d conv2(
        16, 32, std::make_tuple(5, 5), std::make_tuple(1, 1),
        "conv2", true, true, "valid", std::make_tuple(2, 2, 2, 2),
        "zero", "cpu", "xavier"
    );
    CppNet::Activations::ReLU relu2;
    
    CppNet::Layers::MaxPool2D pool2(
        std::make_tuple(2, 2), std::make_tuple(2, 2),
        "pool2", "valid", std::make_tuple(0, 0, 0, 0),
        "zero", "cpu"
    );
        
    CppNet::Layers::Flatten flatten("flatten");

    CppNet::Layers::Linear fc1(
        8192, 128, "fc1", true, true, "cpu", "xavier"  // Adjusted for smaller feature map
    );
    CppNet::Activations::ReLU relu3;

    CppNet::Layers::Linear fc2(
        128, num_classes, "fc2", true, true, "cpu", "xavier"
    );
    CppNet::Activations::Softmax softmax;
    
    // Loss function
    CppNet::Losses::CategoricalCrossEntropy loss("mean", false, 0.0);  // from_logits=false since we use softmax

    // Set number of threads for all layers
    conv1.set_num_threads(num_threads);
    relu1.set_num_threads(num_threads);
    pool1.set_num_threads(num_threads);
    conv2.set_num_threads(num_threads);
    relu2.set_num_threads(num_threads);
    pool2.set_num_threads(num_threads);
    fc1.set_num_threads(num_threads);
    relu3.set_num_threads(num_threads);
    fc2.set_num_threads(num_threads);
    softmax.set_num_threads(num_threads);
    loss.set_num_threads(num_threads);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    double total_loss = 0.0;
    
    for (int epoch = 0; epoch < epochs_to_test; ++epoch) {
        // Forward pass
        auto conv1_out = conv1.forward(input);
        std::cout << "Conv1 output shape: (" << conv1_out.dimension(0) << "," 
                  << conv1_out.dimension(1) << "," 
                  << conv1_out.dimension(2) << "," 
                  << conv1_out.dimension(3) << ")" << std::endl;

        auto relu1_out = relu1.forward(conv1_out);
        auto pool1_out = pool1.forward(relu1_out);
        std::cout << "Pool1 output shape: (" << pool1_out.dimension(0) << "," 
                  << pool1_out.dimension(1) << "," 
                  << pool1_out.dimension(2) << "," 
                  << pool1_out.dimension(3) << ")" << std::endl;
        
        auto conv2_out = conv2.forward(pool1_out);
        std::cout << "Conv2 output shape: (" << conv2_out.dimension(0) << "," 
                  << conv2_out.dimension(1) << "," 
                  << conv2_out.dimension(2) << "," 
                  << conv2_out.dimension(3) << ")" << std::endl;
        auto relu2_out = relu2.forward(conv2_out);
        auto pool2_out = pool2.forward(relu2_out);
        std::cout << "Pool2 output shape: (" << pool2_out.dimension(0) << "," 
                  << pool2_out.dimension(1) << "," 
                  << pool2_out.dimension(2) << "," 
                  << pool2_out.dimension(3) << ")" << std::endl;
        
        auto flatten_out = flatten.forward(pool2_out);
        std::cout << "Flattened size: (" << flatten_out.dimension(0) << "," <<flatten_out.dimension(1) << ")" << std::endl;

        auto fc1_out = fc1.forward(flatten_out);
        std::cout << "FC1 output shape: (" << fc1_out.dimension(0) << "," << fc1_out.dimension(1) << ")" << std::endl;
        auto relu3_out = relu3.forward(fc1_out);
        auto fc2_out = fc2.forward(relu3_out);
        std::cout << "FC2 output shape: (" << fc2_out.dimension(0) << "," << fc2_out.dimension(1) << ")" << std::endl;
        auto predictions = softmax.forward(fc2_out);
        std::cout << "Predictions shape: (" << predictions.dimension(0) << "," << predictions.dimension(1) << ")" << std::endl;
        
        // Compute loss
        double batch_loss = loss.forward(predictions, targets);
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << std::fixed << std::setprecision(4) << batch_loss << std::endl;
        total_loss += batch_loss;
        
        // Backward pass
        auto loss_grad = loss.backward(predictions, targets);
        std::cout << "Loss gradient shape: (" << loss_grad.dimension(0) << "," << loss_grad.dimension(1) << ")" << std::endl;

        auto softmax_grad = softmax.backward(loss_grad);
        std::cout << "Softmax gradient shape: (" << softmax_grad.dimension(0) << "," << softmax_grad.dimension(1) << ")" << std::endl;

        auto fc2_grad = fc2.backward(softmax_grad);
        std::cout << "FC2 gradient shape: (" << fc2_grad.dimension(0) << "," << fc2_grad.dimension(1) << ")" << std::endl;
        auto relu3_grad = relu3.backward(fc2_grad);
        std::cout << "ReLU3 gradient shape: (" << relu3_grad.dimension(0) << "," << relu3_grad.dimension(1) << ")" << std::endl;
        auto fc1_grad = fc1.backward(relu3_grad);
        std::cout << "FC1 gradient shape: (" << fc1_grad.dimension(0) << "," << fc1_grad.dimension(1) << ")" << std::endl;
        auto flatten_grad = flatten.backward(fc1_grad);
        std::cout << "Flatten gradient shape: (" << flatten_grad.dimension(0) << "," << flatten_grad.dimension(1) << ")" << std::endl;
        
        auto pool2_grad = pool2.backward(flatten_grad);
        std::cout << "Pool2 gradient shape: (" << pool2_grad.dimension(0) << "," 
                  << pool2_grad.dimension(1) << "," 
                  << pool2_grad.dimension(2) << "," 
                  << pool2_grad.dimension(3) << ")" << std::endl;
        auto relu2_grad = relu2.backward(pool2_grad);
        std::cout << "ReLU2 gradient shape: (" << relu2_grad.dimension(0) << "," 
                  << relu2_grad.dimension(1) << "," 
                  << relu2_grad.dimension(2) << "," 
                  << relu2_grad.dimension(3) << ")" << std::endl;   
        auto conv2_grad = conv2.backward(relu2_grad);
        std::cout << "Conv2 gradient shape: (" << conv2_grad.dimension(0) << "," 
                  << conv2_grad.dimension(1) << "," 
                  << conv2_grad.dimension(2) << "," 
                  << conv2_grad.dimension(3) << ")" << std::endl;
        
        auto pool1_grad = pool1.backward(conv2_grad);
        std::cout << "Pool1 gradient shape: (" << pool1_grad.dimension(0) << "," 
                  << pool1_grad.dimension(1) << "," 
                  << pool1_grad.dimension(2) << "," 
                  << pool1_grad.dimension(3) << ")" << std::endl;
        auto relu1_grad = relu1.backward(pool1_grad);
        std::cout << "ReLU1 gradient shape: (" << relu1_grad.dimension(0) << "," 
                  << relu1_grad.dimension(1) << "," 
                  << relu1_grad.dimension(2) << "," 
                  << relu1_grad.dimension(3) << ")" << std::endl;
        auto conv1_grad = conv1.backward(relu1_grad);
        std::cout << "Conv1 gradient shape: (" << conv1_grad.dimension(0) << "," 
                  << conv1_grad.dimension(1) << "," 
                  << conv1_grad.dimension(2) << "," 
                  << conv1_grad.dimension(3) << ")" << std::endl;
        
        // Reset gradients for next iteration
        conv1.reset_grads();
        conv2.reset_grads();
        fc1.reset_grads();
        fc2.reset_grads();
    }
    
    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double time_seconds = duration.count() / 1000.0;
    
    double avg_loss = total_loss / epochs_to_test;
    
    std::cout << "Epochs: " << epochs_to_test 
              << ", Avg Loss: " << std::fixed << std::setprecision(4) << avg_loss
              << ", Time: " << std::setprecision(2) << time_seconds << " seconds" << std::endl;
    
    return time_seconds;
}

int main() {
    std::cout << "=== Deep Learning Library Multi-threading Benchmark ===" << std::endl;
    std::cout << "Network: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Flatten -> Linear -> ReLU -> Linear -> Softmax" << std::endl;
    std::cout << "Input: [16, 3, 64, 64], Output: [16, 10]" << std::endl;

    // Test with different number of threads
    std::vector<int> thread_counts = {1, 2, 4, 6, 8};
    std::vector<double> execution_times;
    
    for (int threads : thread_counts) {
        double time = run_training_with_threads(threads, 5);  // 5 epochs for better timing
        execution_times.push_back(time);
    }
    
    // Print summary
    std::cout << "\n=== BENCHMARK RESULTS ===" << std::endl;
    std::cout << std::setw(8) << "Threads" << std::setw(12) << "Time (s)" << std::setw(12) << "Speedup" << std::setw(15) << "Efficiency" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    double baseline_time = execution_times[0];  // Single thread time
    
    for (size_t i = 0; i < thread_counts.size(); ++i) {
        double speedup = baseline_time / execution_times[i];
        double efficiency = speedup / thread_counts[i] * 100.0;
        
        std::cout << std::setw(8) << thread_counts[i] 
                  << std::setw(12) << std::fixed << std::setprecision(2) << execution_times[i]
                  << std::setw(12) << std::setprecision(2) << speedup
                  << std::setw(14) << std::setprecision(1) << efficiency << "%" << std::endl;
    }
    
    return 0;
}
*/




int main()
{
    CppNet::Layers::Flatten flatten("flatten");
    Eigen::Tensor<double, 4> input(2, 3, 4, 4);
    input.setRandom();
    
    std::cout << "Input shape: (" << input.dimension(0) << ","
              << input.dimension(1) << ","
              << input.dimension(2) << ","
              << input.dimension(3) << ")" << std::endl;
    
    auto out = flatten.forward(input);
    std::cout << "Flattened shape: (" << out.dimension(0) << "," << out.dimension(1) << ")" << std::endl;
    
    // Create a proper gradient tensor (same shape as flattened output)
    Eigen::Tensor<double, 2> grad_output(2, 48);
    grad_output.setRandom(); // or setConstant(1.0) for testing
    
    // Test backward pass with proper gradient
    auto grad_input = flatten.backward(grad_output);
    std::cout << "Grad input shape: (" << grad_input.dimension(0) << ","
              << grad_input.dimension(1) << ","
              << grad_input.dimension(2) << ","
              << grad_input.dimension(3) << ")" << std::endl;
    
    // Verify that forward->backward preserves data structure
    // Check if reshaping back gives us the same layout
    int count = 0;
    bool all_match = true;
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 4; ++l) {
                    // Compare grad_input with corresponding grad_output values
                    if (grad_input(i, j, k, l) != grad_output(i, count)) {
                        std::cout << "Mismatch at (" << i << "," << j << "," << k << "," << l << "): "
                                  << grad_input(i, j, k, l) << " != " << grad_output(i, count) << std::endl;
                        all_match = false;
                    }
                    count++;
                }
            }
        }
    }
    
    if (all_match) {
        std::cout << "SUCCESS: All values match! Flatten backward works correctly." << std::endl;
    } else {
        std::cout << "FAILED: Some values don't match." << std::endl;
    }
    
    return 0;
}