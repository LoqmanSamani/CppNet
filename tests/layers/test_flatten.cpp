/**
 * @file test_flatten.cpp
 * @brief Unit tests for the Flatten layer
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/flatten.hpp"

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
    std::cout << "=== Flatten Layer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Forward - output shape [batch, C*H*W]")
    {
        CppNet::Layers::Flatten flatten;
        Eigen::Tensor<float, 4> input(2, 3, 4, 5);
        input.setRandom();
        auto output = flatten.forward(input);

        ASSERT_TRUE(output.dimension(0) == 2);
        ASSERT_TRUE(output.dimension(1) == 3 * 4 * 5);  // 60
    }
    END_TEST("Forward - output shape [batch, C*H*W]");

    // ---------------------------------------------------------------
    TEST("Forward - data preserved")
    {
        CppNet::Layers::Flatten flatten;
        Eigen::Tensor<float, 4> input(1, 2, 2, 2);
        input.setValues({{{{1.0f, 2.0f}, {3.0f, 4.0f}},
                          {{5.0f, 6.0f}, {7.0f, 8.0f}}}});
        auto output = flatten.forward(input);

        ASSERT_TRUE(output.dimension(0) == 1);
        ASSERT_TRUE(output.dimension(1) == 8);

        // All original values should be present in the flat tensor
        float total_in = 0.0f, total_out = 0.0f;
        for (int i = 0; i < input.size(); ++i)
            total_in += input.data()[i];
        for (int j = 0; j < output.dimension(1); ++j)
            total_out += output(0, j);
        ASSERT_NEAR(total_in, total_out, 1e-5f);
    }
    END_TEST("Forward - data preserved");

    // ---------------------------------------------------------------
    TEST("Backward - restores original 4D shape")
    {
        CppNet::Layers::Flatten flatten;
        Eigen::Tensor<float, 4> input(2, 3, 4, 5);
        input.setRandom();
        auto output = flatten.forward(input);

        Eigen::Tensor<float, 2> grad_output(2, 60);
        grad_output.setConstant(1.0f);
        auto grad_input = flatten.backward(grad_output);

        ASSERT_TRUE(grad_input.dimension(0) == 2);
        ASSERT_TRUE(grad_input.dimension(1) == 3);
        ASSERT_TRUE(grad_input.dimension(2) == 4);
        ASSERT_TRUE(grad_input.dimension(3) == 5);
    }
    END_TEST("Backward - restores original 4D shape");

    // ---------------------------------------------------------------
    TEST("Backward - data preserved in reshape")
    {
        CppNet::Layers::Flatten flatten;
        Eigen::Tensor<float, 4> input(1, 1, 2, 3);
        input.setRandom();
        flatten.forward(input);

        Eigen::Tensor<float, 2> grad_output(1, 6);
        grad_output.setRandom();
        auto grad_input = flatten.backward(grad_output);

        float total_grad_out = 0.0f, total_grad_in = 0.0f;
        for (int j = 0; j < 6; ++j)
            total_grad_out += grad_output(0, j);
        for (int i = 0; i < grad_input.size(); ++i)
            total_grad_in += grad_input.data()[i];
        ASSERT_NEAR(total_grad_out, total_grad_in, 1e-5f);
    }
    END_TEST("Backward - data preserved in reshape");

    // ---------------------------------------------------------------
    TEST("Not trainable")
    {
        CppNet::Layers::Flatten flatten;
        ASSERT_TRUE(!flatten.is_trainable());
    }
    END_TEST("Not trainable");

    // ---------------------------------------------------------------
    TEST("Single channel single spatial")
    {
        CppNet::Layers::Flatten flatten;
        Eigen::Tensor<float, 4> input(4, 1, 1, 1);
        input.setValues({{{{1.0f}}}, {{{2.0f}}}, {{{3.0f}}}, {{{4.0f}}}});
        auto output = flatten.forward(input);

        ASSERT_TRUE(output.dimension(0) == 4);
        ASSERT_TRUE(output.dimension(1) == 1);
        ASSERT_NEAR(output(0, 0), 1.0f, 1e-6f);
        ASSERT_NEAR(output(1, 0), 2.0f, 1e-6f);
        ASSERT_NEAR(output(2, 0), 3.0f, 1e-6f);
        ASSERT_NEAR(output(3, 0), 4.0f, 1e-6f);
    }
    END_TEST("Single channel single spatial");

    // ---------------------------------------------------------------
    TEST("Round-trip forward then backward preserves values")
    {
        CppNet::Layers::Flatten flatten;
        Eigen::Tensor<float, 4> input(2, 2, 3, 3);
        input.setRandom();
        auto output = flatten.forward(input);
        auto reconstructed = flatten.backward(output);

        for (int i = 0; i < input.size(); ++i)
            ASSERT_NEAR(input.data()[i], reconstructed.data()[i], 1e-6f);
    }
    END_TEST("Round-trip forward then backward preserves values");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
