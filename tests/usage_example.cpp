/**
 * @file example_usage.cpp
 * @brief Example showing how to use CppNet library
 * 
 * Compile this after installing CppNet:
 *   g++ -std=c++17 example_usage.cpp -lCppNet -lgomp -o example
 * 
 * Or with CMake:
 *   find_package(CppNet REQUIRED)
 *   target_link_libraries(your_app CppNet::CppNet)
 */

#include <CppNet/CppNet.hpp>
#include <iostream>

int main() {
    // Print library information
    std::cout << "CppNet Version: " << CppNet::version() << std::endl;
    std::cout << "CUDA Support: " << (CppNet::has_cuda_support() ? "Yes" : "No") << std::endl;
    std::cout << "OpenMP Support: " << (CppNet::has_openmp_support() ? "Yes" : "No") << std::endl;
    std::cout << std::endl;

    // Create a simple neural network
    std::cout << "Creating a neural network..." << std::endl;
    
    // Input layer: 784 features (28x28 image)
    // Hidden layer: 128 neurons
    // Output layer: 10 classes
    
    auto layer1 = std::make_shared<CppNet::Layers::Linear>(
        784,              // input size
        128,              // output size
        "hidden_layer",   // name
        true,             // trainable
        true,             // use bias
        "cpu",            // device
        "he"              // weight initialization
    );
    
    auto layer2 = std::make_shared<CppNet::Layers::Linear>(
        128,              // input size
        10,               // output size
        "output_layer",   // name
        true,             // trainable
        true,             // use bias
        "cpu",            // device
        "xavier"          // weight initialization
    );
    
    // Print layer information
    std::cout << "\nLayer 1 Information:" << std::endl;
    layer1->print_layer_info();
    
    std::cout << "\nLayer 2 Information:" << std::endl;
    layer2->print_layer_info();
    
    // Create a sample input (batch_size=2, features=784)
    Eigen::Tensor<float, 2> input(2, 784);
    input.setRandom();
    
    std::cout << "\nInput shape: [" << input.dimension(0) 
              << ", " << input.dimension(1) << "]" << std::endl;
    
    // Forward pass through layer 1
    auto hidden = layer1->forward(input);
    std::cout << "Hidden layer output shape: [" << hidden.dimension(0) 
              << ", " << hidden.dimension(1) << "]" << std::endl;
    
    // Apply ReLU activation
    CppNet::Activations::ReLU relu;
    auto activated = relu.forward(hidden);
    
    // Forward pass through layer 2
    auto output = layer2->forward(activated);
    std::cout << "Final output shape: [" << output.dimension(0) 
              << ", " << output.dimension(1) << "]" << std::endl;
    
    // Apply Sigmoid activation for binary classification
    // or Softmax for multi-class (not shown here)
    CppNet::Activations::Sigmoid sigmoid;
    auto predictions = sigmoid.forward(output);
    
    std::cout << "\nForward pass completed successfully!" << std::endl;
    
    // Create optimizer
    CppNet::Optimizers::SGD optimizer(0.01);  // learning rate = 0.01
    
    std::cout << "\nNetwork ready for training!" << std::endl;
    
    return 0;
}