#include "layers.hpp"
#include <iostream>

int main() {
    std::cout << "Testing CppNet layers compilation..." << std::endl;
    
    // Test basic instantiation
    CppNet::Layers::Linear linear;
    CppNet::Layers::Conv2d conv;
    CppNet::Layers::Flatten flatten;
    
    // Test with dummy tensors
    Eigen::Tensor<double, 2> dummy_2d(2, 3);
    dummy_2d.setRandom();
    
    Eigen::Tensor<double, 4> dummy_4d(1, 3, 4, 4);
    dummy_4d.setRandom();
    
    // Test forward passes (they'll return zeros for now)
    auto linear_out = linear.forward(dummy_2d);
    auto conv_out = conv.forward(dummy_4d);
    auto flatten_out = flatten.forward(dummy_4d);
    
    std::cout << "Linear output dimensions: " << linear_out.dimension(0) << "x" << linear_out.dimension(1) << std::endl;
    std::cout << "Conv2d output dimensions: " << conv_out.dimension(0) << "x" << conv_out.dimension(1) 
              << "x" << conv_out.dimension(2) << "x" << conv_out.dimension(3) << std::endl;
    std::cout << "Flatten output dimensions: " << flatten_out.dimension(0) << "x" << flatten_out.dimension(1) << std::endl;
    
    std::cout << "All tests passed! Library skeleton is compilable." << std::endl;
    return 0;
}