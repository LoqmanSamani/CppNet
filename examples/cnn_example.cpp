#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <random>

int main() {
    // Start timing
    //auto start = std::chrono::high_resolution_clock::now();
    
    Eigen::Tensor<double, 4> input1(10, 3, 100, 100);
    input1.setRandom();

    Eigen::Tensor<double, 2> y(10, 10);  // 10 samples, 10 classes
    y.setZero();  // Initialize with zeros

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 9);

    // Create one-hot encoded labels
    for (int i = 0; i < 10; ++i) 
    {
        int class_label = dis(gen);
        y(i, class_label) = 1.0;  // Set the correct class to 1
    }
    
    std::tuple<int, int> kernel_size = std::make_tuple(7, 7);
    std::tuple<int, int> stride = std::make_tuple(1, 1);
    std::tuple<int, int> stride1 = std::make_tuple(2, 2);
    std::tuple<int, int, int, int> num_padding = std::make_tuple(2, 2, 2, 2);
    std::tuple<int, int, int, int> num_padding1 = std::make_tuple(0, 0, 0, 0);
    int in_size1 = 7220;
    int out_size1 = 200;
    int in_size2 = 200;
    int out_size2 = 10;
    
    // int in_channels,
    // int out_channels,
    // Activations::Activation* activator = nullptr,
    // std::tuple<int, int> kernel_size = std::make_tuple(3, 3),
    // std::tuple<int, int> stride = std::make_tuple(1, 1),
    // std::string padding = "valid",
    // std::tuple<int, int, int, int> num_padding = std::make_tuple(0, 0, 0, 0),
    // std::string padding_mode = "zero",
    // std::string layer_name = "Conv2D",
    // bool trainable = true,
    // bool bias = true
    CppNet::Layers::Conv2d conv1(3, 10, nullptr, kernel_size, stride, "valid", num_padding, "zero", "Conv2D1", true, false);
    CppNet::Layers::Conv2d conv2(10, 20, nullptr, kernel_size, stride, "valid", num_padding, "zero", "Conv2D2", true, false);

    // std::tuple<int, int> kernel_size = std::make_tuple(3, 3),
    // std::tuple<int, int> stride = std::make_tuple(1, 1),
    // std::string padding = "valid",
    // std::tuple<int, int, int, int> num_padding = std::make_tuple(0, 0, 0, 0),
    // std::string padding_mode = "zero",
    // std::string layer_name = "MaxPool2D"

    CppNet::Layers::MaxPool2D pool1(kernel_size, stride1, "valid", num_padding1, "zero", "Maxpool1");
    CppNet::Layers::MaxPool2D pool2(kernel_size, stride1, "valid", num_padding1, "zero", "Maxpool2");
    
    //int start_dim = 1, 
    //int end_dim = -1,  
    //std::string layer_name = "Flatten"
    CppNet::Layers::Flatten flatten(1, -1, "Flatten");

    // int in_size, 
    // int out_size, 
    // std::string layer_name = "Linear", 
    // bool trainable = true, 
    // bool bias = true
    CppNet::Layers::Linear dense1(in_size1, out_size1, "Dense1", true, true);
    CppNet::Layers::Linear dense2(in_size2, out_size2, "Dense2", true, true);

    CppNet::Activations::ReLU relu1;
    CppNet::Activations::ReLU relu2;
    CppNet::Activations::SoftMax softmax;

    CppNet::Optimizers::SGD optimizer;
    CppNet::Losses::CategoricalCrossEntropy loss_fn;
    double lr = 0.001;
    
    
    for (int i = 1; i < 6; i++) 
    {
        // Forward pass
        Eigen::Tensor<double, 4> conv_1 = conv1.forward(input1);
        //std::cout << "conv 1: " << conv_1.dimensions() << std::endl;
        Eigen::Tensor<double, 4> pool_1 = pool1.forward(conv_1);
        //std::cout << "pool1: " << pool_1.dimensions() << std::endl;
        Eigen::Tensor<double, 4> conv_2 = conv2.forward(pool_1);
        //std::cout << "conv 2: " << conv_2.dimensions() << std::endl;
        Eigen::Tensor<double, 4> pool_2 = pool2.forward(conv_2);
        //std::cout << "pool2: " << pool_2.dimensions() << std::endl;
        Eigen::Tensor<double, 2> flattened = flatten.forward(pool_2);
        std::cout << "flattened: " << flattened.dimensions() << std::endl;
        Eigen::Tensor<double, 2> dense_1 = dense1.forward(flattened);
        Eigen::Tensor<double, 2> relu_1 = relu1.forward(dense_1);
        //std::cout << "dense 1: " << dense_1.dimensions() << std::endl;
        Eigen::Tensor<double, 2> dense_2 = dense2.forward(relu_1);
        Eigen::Tensor<double, 2> relu_2 = relu2.forward(dense_2);
        //std::cout << "dense 2: " << dense_2.dimensions() << std::endl;
        //std::cout << "relu_2 dimensions: " << relu_2.dimensions() << std::endl;
        Eigen::Tensor<double, 2> softmax_ = softmax.forward(relu_2);
        //std::cout << "softmax: " << softmax_.dimensions() << std::endl;
        
        // Loss
        double loss = loss_fn.forward(softmax_, y);
        //std::cout << "Loss: " << loss << std::endl;

        // reset gradients
        conv1.reset_grads();
        conv2.reset_grads();
        dense1.reset_grads();
        dense2.reset_grads();
        
        // Backward pass with shape debugging
        Eigen::Tensor<double, 2> grad_out = loss_fn.backward(y, softmax_);
        //std::cout << "grad_out: " << grad_out.dimensions() << std::endl;
        Eigen::Tensor<double, 2> relu2_b = relu2.backward(grad_out);
        //std::cout << "relu2_b: " << relu2_b.dimensions() << std::endl;
        Eigen::Tensor<double, 2> dense2_b = dense2.backward(relu2_b);
        //std::cout << "dense2_b: " << dense2_b.dimensions() << std::endl;
        Eigen::Tensor<double, 2> relu1_b = relu1.backward(dense2_b);
        //std::cout << "relu1_b: " << relu1_b.dimensions() << std::endl;
        Eigen::Tensor<double, 2> dense1_b = dense1.backward(relu1_b);
        //std::cout << "dense1_b: " << dense1_b.dimensions() << std::endl;
        Eigen::Tensor<double, 4> flattened_b = flatten.backward4D(dense1_b);
        //std::cout << "flattened_b: " << flattened_b.dimensions() << std::endl;
        Eigen::Tensor<double, 4> pool2_b = pool2.backward(flattened_b);
        //std::cout << "pool2_b: " << pool2_b.dimensions() << std::endl;
        Eigen::Tensor<double, 4> conv2_b = conv2.backward(pool2_b);
        //std::cout << "conv2_b: " << conv2_b.dimensions() << std::endl;
        Eigen::Tensor<double, 4> pool1_b = pool1.backward(conv2_b);
        //std::cout << "pool1_b: " << pool1_b.dimensions() << std::endl;
        Eigen::Tensor<double, 4> conv1_b = conv1.backward(pool1_b);
        //std::cout << "conv1_b: " << conv1_b.dimensions() << std::endl;

        std::cout << "Iteration " << i << "| Loss: " << loss << std::endl;

        // update parameters
        conv1.update_parameters(optimizer, lr);
        conv2.update_parameters(optimizer, lr);
        dense1.update_parameters(optimizer, lr);
        dense2.update_parameters(optimizer, lr);
        
        /*
        if (i == 1)
        {
            return 0;
        }
        */
        
    }
    
    return 0;
}