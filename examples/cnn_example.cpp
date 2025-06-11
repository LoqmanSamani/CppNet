#include "layers.hpp"
#include "optimizers.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>



int main() {
    Eigen::Tensor<double, 4> input_(2, 3, 100, 100);
    Eigen::Tensor<double, 4> output_(2, 20, 96, 96);
    input_.setRandom();
    output_.setRandom();

    std::tuple<int, int> kernel_size = std::make_tuple(7, 7);
    std::tuple<int, int> stride = std::make_tuple(3, 3);
    std::tuple<int, int, int, int> num_padding = std::make_tuple(2, 2, 2, 2);

     

    CppNet::Layers::Conv2d conv(3, 12, nullptr, kernel_size, stride, "valid", num_padding, "zero", "conv", true, true);

    Eigen::Tensor<double, 4> forw = conv.forward(input_);
    Eigen::Tensor<double, 4> back = conv.backward(forw);

    std::cout << forw.dimensions() << std::endl;
    std::cout << back.dimensions() << std::endl;


    return 0;
}