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



// Function to run training with specific number of threads and return timing results
double run_training_with_threads(int num_threads, int epochs_to_test = 1) {
    std::cout << "\n=== Testing with " << num_threads << " threads ===" << std::endl;
    
    // Data
    Eigen::Tensor<double, 4> input(20, 3, 128, 128);
    input.setRandom();

    // layers
    CppNet::Layers::Conv2d conv1(
        3, 10, std::make_tuple(3, 3), std::make_tuple(1, 1),
        "conv 1", true, true, "valid", std::make_tuple(2, 2, 2, 2),
        "zero", "cpu", "xavier"
        );
    CppNet::Layers::Conv2d conv2(
        10, 20, std::make_tuple(3, 3), std::make_tuple(1, 1),
        "conv 2", true, true, "valid", std::make_tuple(2, 2, 2, 2),
        "zero", "cpu", "xavier"
        );
    CppNet::Layers::Conv2d conv3(
        20, 30, std::make_tuple(3, 3), std::make_tuple(1, 1),
        "conv 3", true, true, "valid", std::make_tuple(2, 2, 2, 2),
        "zero", "cpu", "xavier"
        );  
    CppNet::Layers::Conv2d conv4(
        30, 40, std::make_tuple(3, 3), std::make_tuple(1, 1),
        "conv 4", true, true, "valid", std::make_tuple(2, 2, 2, 2),
        "zero", "cpu", "xavier"
        );  
    
    // activations
    CppNet::Activations::ReLU act1;
    CppNet::Activations::Sigmoid act2;
    CppNet::Activations::ReLU act3;
    CppNet::Activations::Sigmoid act4;

    // Set number of threads for all layers
    conv1.set_num_threads(num_threads);
    conv2.set_num_threads(num_threads);
    conv3.set_num_threads(num_threads);
    conv4.set_num_threads(num_threads);

    // Start timing the training loop
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < epochs_to_test; ++i)
    {
        // Forward pass
        Eigen::Tensor<double, 4> out1 = conv1.forward(input);
        Eigen::Tensor<double, 4> act_out1 = act1.forward(out1);
        Eigen::Tensor<double, 4> out2 = conv2.forward(act_out1);
        Eigen::Tensor<double, 4> act_out2 = act2.forward(out2);
        Eigen::Tensor<double, 4> out3 = conv3.forward(act_out2);
        Eigen::Tensor<double, 4> act_out3 = act3.forward(out3);
        Eigen::Tensor<double, 4> out4 = conv4.forward(act_out3);
        Eigen::Tensor<double, 4> act_out4 = act4.forward(out4);

        // Reset gradients
        conv1.reset_grads();
        conv2.reset_grads();
        conv3.reset_grads();
        conv4.reset_grads();    

        // Backward pass
        Eigen::Tensor<double, 4> grad_output = act4.backward(act_out4);
        Eigen::Tensor<double, 4> grad_conv4 = conv4.backward(grad_output);
        Eigen::Tensor<double, 4> grad_act3 = act3.backward(grad_conv4);
        Eigen::Tensor<double, 4> grad_conv3 = conv3.backward(grad_act3);
        Eigen::Tensor<double, 4> grad_act2 = act2.backward(grad_conv3);
        Eigen::Tensor<double, 4> grad_conv2 = conv2.backward(grad_act2);
        Eigen::Tensor<double, 4> grad_act1 = act1.backward(grad_conv2);
        Eigen::Tensor<double, 4> grad_conv1 = conv1.backward(grad_act1);
    }
    
    
    // Stop timing and calculate duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double time_seconds = duration.count() / 1000.0;
    
    std::cout << "Training completed in: " << std::fixed << std::setprecision(2) 
              << time_seconds << " seconds" << std::endl;
    
    return time_seconds;
}





int main(){

    // Test with different number of threads
    std::vector<int> thread_counts = {1, 2, 4, 6, 8};
    for (int threads : thread_counts) { 
        run_training_with_threads(threads);
    }   
    return 0;
}


