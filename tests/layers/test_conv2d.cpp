/**
 * @file test_conv2d.cpp
 * @brief Unit tests for the Conv2D layer
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/conv2d.hpp"
#include "CppNet/optimizers/sgd.hpp"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                                           \
    do {                                                                     \
        std::cout << "  [TEST] " << name << " ... ";                         \
        try {

#define END_TEST(name)                                                       \
            std::cout << "PASSED" << std::endl;                              \
            ++tests_passed;                                                  \
        } catch (const std::exception& e) {                                  \
            std::cout << "FAILED: " << e.what() << std::endl;                \
            ++tests_failed;                                                  \
        } catch (...) {                                                      \
            std::cout << "FAILED (unknown exception)" << std::endl;          \
            ++tests_failed;                                                  \
        }                                                                    \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    if (std::fabs((a) - (b)) > (eps)) \
        throw std::runtime_error( \
            std::string("Expected ") + std::to_string(b) + " but got " + std::to_string(a))

#define ASSERT_TRUE(cond) \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond)

int main()
{
    std::cout << "=== Conv2D Layer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Construction - basic parameters")
    {
        CppNet::Layers::Conv2D conv(3, 16, 3, 1, 0, true, "cpu-eigen");
        ASSERT_TRUE(conv.get_in_channels() == 3);
        ASSERT_TRUE(conv.get_out_channels() == 16);
        ASSERT_TRUE(conv.get_kernel_size() == 3);
        ASSERT_TRUE(conv.get_stride() == 1);
        ASSERT_TRUE(conv.get_padding() == 0);
        ASSERT_TRUE(conv.is_trainable());
    }
    END_TEST("Construction - basic parameters");

    // ---------------------------------------------------------------
    TEST("Weight shape is [out_ch, in_ch, kH, kW]")
    {
        CppNet::Layers::Conv2D conv(3, 8, 5, 1, 0, true, "cpu-eigen");
        auto& w = conv.get_weights();
        ASSERT_TRUE(w.dimension(0) == 8);
        ASSERT_TRUE(w.dimension(1) == 3);
        ASSERT_TRUE(w.dimension(2) == 5);
        ASSERT_TRUE(w.dimension(3) == 5);
    }
    END_TEST("Weight shape is [out_ch, in_ch, kH, kW]");

    // ---------------------------------------------------------------
    TEST("Bias shape is [out_ch]")
    {
        CppNet::Layers::Conv2D conv(3, 8, 3, 1, 0, true, "cpu-eigen");
        auto& b = conv.get_biases();
        ASSERT_TRUE(b.dimension(0) == 8);
    }
    END_TEST("Bias shape is [out_ch]");

    // ---------------------------------------------------------------
    TEST("Forward - output shape no padding")
    {
        // Input: [batch=1, channels=1, H=8, W=8], kernel=3, stride=1, pad=0
        // Output H = (8 - 3)/1 + 1 = 6
        CppNet::Layers::Conv2D conv(1, 4, 3, 1, 0, true, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 8, 8);
        input.setRandom();
        auto output = conv.forward(input);

        ASSERT_TRUE(output.dimension(0) == 1);
        ASSERT_TRUE(output.dimension(1) == 4);
        ASSERT_TRUE(output.dimension(2) == 6);
        ASSERT_TRUE(output.dimension(3) == 6);
    }
    END_TEST("Forward - output shape no padding");

    // ---------------------------------------------------------------
    TEST("Forward - output shape with padding")
    {
        // Input: [1, 1, 8, 8], kernel=3, stride=1, pad=1
        // Output H = (8 + 2*1 - 3)/1 + 1 = 8
        CppNet::Layers::Conv2D conv(1, 4, 3, 1, 1, true, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 8, 8);
        input.setRandom();
        auto output = conv.forward(input);

        ASSERT_TRUE(output.dimension(0) == 1);
        ASSERT_TRUE(output.dimension(1) == 4);
        ASSERT_TRUE(output.dimension(2) == 8);
        ASSERT_TRUE(output.dimension(3) == 8);
    }
    END_TEST("Forward - output shape with padding");

    // ---------------------------------------------------------------
    TEST("Forward - output shape with stride")
    {
        // Input: [1, 3, 16, 16], kernel=3, stride=2, pad=0
        // Output H = (16 - 3)/2 + 1 = 7
        CppNet::Layers::Conv2D conv(3, 8, 3, 2, 0, true, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 3, 16, 16);
        input.setRandom();
        auto output = conv.forward(input);

        ASSERT_TRUE(output.dimension(0) == 1);
        ASSERT_TRUE(output.dimension(1) == 8);
        ASSERT_TRUE(output.dimension(2) == 7);
        ASSERT_TRUE(output.dimension(3) == 7);
    }
    END_TEST("Forward - output shape with stride");

    // ---------------------------------------------------------------
    TEST("Forward - batch dimension preserved")
    {
        CppNet::Layers::Conv2D conv(1, 4, 3, 1, 0, true, "cpu-eigen");
        Eigen::Tensor<float, 4> input(4, 1, 8, 8);
        input.setRandom();
        auto output = conv.forward(input);

        ASSERT_TRUE(output.dimension(0) == 4);
    }
    END_TEST("Forward - batch dimension preserved");

    // ---------------------------------------------------------------
    TEST("Forward - output is finite")
    {
        CppNet::Layers::Conv2D conv(1, 2, 3, 1, 0, true, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 5, 5);
        input.setRandom();
        auto output = conv.forward(input);

        for (int b = 0; b < output.dimension(0); ++b)
            for (int c = 0; c < output.dimension(1); ++c)
                for (int h = 0; h < output.dimension(2); ++h)
                    for (int w = 0; w < output.dimension(3); ++w)
                        ASSERT_TRUE(std::isfinite(output(b, c, h, w)));
    }
    END_TEST("Forward - output is finite");

    // ---------------------------------------------------------------
    TEST("Backward - grad input shape matches input")
    {
        CppNet::Layers::Conv2D conv(3, 8, 3, 1, 1, true, "cpu-eigen");
        Eigen::Tensor<float, 4> input(2, 3, 8, 8);
        input.setRandom();
        auto output = conv.forward(input);

        Eigen::Tensor<float, 4> grad_output(output.dimensions());
        grad_output.setConstant(1.0f);
        auto grad_input = conv.backward(grad_output);

        ASSERT_TRUE(grad_input.dimension(0) == 2);
        ASSERT_TRUE(grad_input.dimension(1) == 3);
        ASSERT_TRUE(grad_input.dimension(2) == 8);
        ASSERT_TRUE(grad_input.dimension(3) == 8);
    }
    END_TEST("Backward - grad input shape matches input");

    // ---------------------------------------------------------------
    TEST("Backward - grad weights shape")
    {
        CppNet::Layers::Conv2D conv(3, 8, 3, 1, 0, true, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 3, 8, 8);
        input.setRandom();
        auto output = conv.forward(input);

        Eigen::Tensor<float, 4> grad_output(output.dimensions());
        grad_output.setConstant(1.0f);
        conv.backward(grad_output);

        auto& gw = conv.get_grad_weights();
        ASSERT_TRUE(gw.dimension(0) == 8);
        ASSERT_TRUE(gw.dimension(1) == 3);
        ASSERT_TRUE(gw.dimension(2) == 3);
        ASSERT_TRUE(gw.dimension(3) == 3);
    }
    END_TEST("Backward - grad weights shape");

    // ---------------------------------------------------------------
    TEST("Freeze/Unfreeze")
    {
        CppNet::Layers::Conv2D conv(1, 4, 3, 1, 0, true, "cpu-eigen");
        ASSERT_TRUE(conv.is_trainable());
        conv.freeze();
        ASSERT_TRUE(!conv.is_trainable());
        conv.unfreeze();
        ASSERT_TRUE(conv.is_trainable());
    }
    END_TEST("Freeze/Unfreeze");

    // ---------------------------------------------------------------
    TEST("Kernel size 1 convolution")
    {
        CppNet::Layers::Conv2D conv(3, 8, 1, 1, 0, true, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 3, 4, 4);
        input.setRandom();
        auto output = conv.forward(input);

        // With k=1, s=1, p=0: output spatial dims = input spatial dims
        ASSERT_TRUE(output.dimension(2) == 4);
        ASSERT_TRUE(output.dimension(3) == 4);
    }
    END_TEST("Kernel size 1 convolution");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
