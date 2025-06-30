#include "layers.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>


using namespace Eigen;

int main() {
    // Create a 2x3 Tensor
    Tensor<float, 2> tensor2d(2, 3);
    tensor2d.setValues({{1, 2, 3}, {4, 5, 6}});

    // Reshape it to a 3x2 Tensor
    array<Index, 2> new_dims = {3, 2};
    Tensor<float, 2> reshaped_tensor = tensor2d.reshape(new_dims);

    // Print reshaped tensor
    for (int i = 0; i < reshaped_tensor.dimension(0); ++i) {
        for (int j = 0; j < reshaped_tensor.dimension(1); ++j) {
            std::cout << reshaped_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
