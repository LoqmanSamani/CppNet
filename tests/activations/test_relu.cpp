/**
 * @file test_relu.cpp
 * @brief Unit tests for the ReLU activation function
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/activations/relu.hpp"

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
    std::cout << "=== ReLU Activation Tests ===" << std::endl;

    // ---------------------------------------------------------------
    // 2D forward: positive values pass through
    // ---------------------------------------------------------------
    TEST("2D forward - positive values pass through")
    {
        CppNet::Activations::ReLU relu("cpu-eigen");
        Eigen::Tensor<float, 2> input(2, 3);
        input.setValues({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

        auto output = relu.forward(input);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                ASSERT_NEAR(output(i, j), input(i, j), 1e-6f);
    }
    END_TEST("2D forward - positive values pass through");

    // ---------------------------------------------------------------
    // 2D forward: negative values become zero
    // ---------------------------------------------------------------
    TEST("2D forward - negative values become zero")
    {
        CppNet::Activations::ReLU relu("cpu-eigen");
        Eigen::Tensor<float, 2> input(2, 3);
        input.setValues({{-1.0f, -2.0f, -3.0f}, {-0.5f, -100.0f, -0.001f}});

        auto output = relu.forward(input);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                ASSERT_NEAR(output(i, j), 0.0f, 1e-6f);
    }
    END_TEST("2D forward - negative values become zero");

    // ---------------------------------------------------------------
    // 2D forward: mixed positive and negative
    // ---------------------------------------------------------------
    TEST("2D forward - mixed values")
    {
        CppNet::Activations::ReLU relu("cpu-eigen");
        Eigen::Tensor<float, 2> input(2, 3);
        input.setValues({{-1.0f, 2.0f, -3.0f}, {4.0f, -5.0f, 0.0f}});

        auto output = relu.forward(input);
        ASSERT_NEAR(output(0, 0), 0.0f, 1e-6f);
        ASSERT_NEAR(output(0, 1), 2.0f, 1e-6f);
        ASSERT_NEAR(output(0, 2), 0.0f, 1e-6f);
        ASSERT_NEAR(output(1, 0), 4.0f, 1e-6f);
        ASSERT_NEAR(output(1, 1), 0.0f, 1e-6f);
        ASSERT_NEAR(output(1, 2), 0.0f, 1e-6f);
    }
    END_TEST("2D forward - mixed values");

    // ---------------------------------------------------------------
    // 2D backward: gradient flows through positive, blocked for negative
    // ---------------------------------------------------------------
    TEST("2D backward - gradient gating")
    {
        CppNet::Activations::ReLU relu("cpu-eigen");
        Eigen::Tensor<float, 2> input(2, 3);
        input.setValues({{-1.0f, 2.0f, -3.0f}, {4.0f, -5.0f, 0.0f}});
        relu.forward(input);

        Eigen::Tensor<float, 2> grad(2, 3);
        grad.setConstant(1.0f);
        auto grad_input = relu.backward(grad);

        // Where output > 0, grad passes through (multiplied by cached output cwiseMax(0))
        ASSERT_NEAR(grad_input(0, 0), 0.0f, 1e-6f);   // input was negative
        ASSERT_TRUE(grad_input(0, 1) > 0.0f);           // input was positive
        ASSERT_NEAR(grad_input(0, 2), 0.0f, 1e-6f);   // input was negative
        ASSERT_TRUE(grad_input(1, 0) > 0.0f);           // input was positive
        ASSERT_NEAR(grad_input(1, 1), 0.0f, 1e-6f);   // input was negative
    }
    END_TEST("2D backward - gradient gating");

    // ---------------------------------------------------------------
    // 4D forward: works with batch/channel/height/width tensors
    // ---------------------------------------------------------------
    TEST("4D forward - basic operation")
    {
        CppNet::Activations::ReLU relu("cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 2, 2);
        input.setValues({{{{-1.0f, 2.0f}, {3.0f, -4.0f}}}});

        auto output = relu.forward(input);
        ASSERT_NEAR(output(0, 0, 0, 0), 0.0f, 1e-6f);
        ASSERT_NEAR(output(0, 0, 0, 1), 2.0f, 1e-6f);
        ASSERT_NEAR(output(0, 0, 1, 0), 3.0f, 1e-6f);
        ASSERT_NEAR(output(0, 0, 1, 1), 0.0f, 1e-6f);
    }
    END_TEST("4D forward - basic operation");

    // ---------------------------------------------------------------
    // 4D backward
    // ---------------------------------------------------------------
    TEST("4D backward - gradient gating")
    {
        CppNet::Activations::ReLU relu("cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 2, 2);
        input.setValues({{{{-1.0f, 2.0f}, {3.0f, -4.0f}}}});
        relu.forward(input);

        Eigen::Tensor<float, 4> grad(1, 1, 2, 2);
        grad.setConstant(1.0f);
        auto grad_input = relu.backward(grad);

        ASSERT_NEAR(grad_input(0, 0, 0, 0), 0.0f, 1e-6f);
        ASSERT_TRUE(grad_input(0, 0, 0, 1) > 0.0f);
        ASSERT_TRUE(grad_input(0, 0, 1, 0) > 0.0f);
        ASSERT_NEAR(grad_input(0, 0, 1, 1), 0.0f, 1e-6f);
    }
    END_TEST("4D backward - gradient gating");

    // ---------------------------------------------------------------
    // CPU backend
    // ---------------------------------------------------------------
    TEST("CPU backend - forward matches cpu-eigen")
    {
        CppNet::Activations::ReLU relu_cpu("cpu");
        Eigen::Tensor<float, 2> input(3, 4);
        input.setValues({{-1.0f, 0.5f, -0.3f, 2.0f},
                         {0.0f, -2.0f, 1.0f, -0.1f},
                         {3.0f, -3.0f, 0.1f, -0.01f}});

        auto output = relu_cpu.forward(input);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                ASSERT_NEAR(output(i, j), std::max(0.0f, input(i, j)), 1e-6f);
    }
    END_TEST("CPU backend - forward matches cpu-eigen");

    // ---------------------------------------------------------------
    // Output shape matches input shape
    // ---------------------------------------------------------------
    TEST("Output shape matches input shape (2D)")
    {
        CppNet::Activations::ReLU relu("cpu-eigen");
        Eigen::Tensor<float, 2> input(5, 10);
        input.setRandom();
        auto output = relu.forward(input);
        ASSERT_TRUE(output.dimension(0) == 5);
        ASSERT_TRUE(output.dimension(1) == 10);
    }
    END_TEST("Output shape matches input shape (2D)");

    TEST("Output shape matches input shape (4D)")
    {
        CppNet::Activations::ReLU relu("cpu-eigen");
        Eigen::Tensor<float, 4> input(2, 3, 4, 5);
        input.setRandom();
        auto output = relu.forward(input);
        ASSERT_TRUE(output.dimension(0) == 2);
        ASSERT_TRUE(output.dimension(1) == 3);
        ASSERT_TRUE(output.dimension(2) == 4);
        ASSERT_TRUE(output.dimension(3) == 5);
    }
    END_TEST("Output shape matches input shape (4D)");

    // ---------------------------------------------------------------
    // All zeros input
    // ---------------------------------------------------------------
    TEST("All zeros input produces all zeros output")
    {
        CppNet::Activations::ReLU relu("cpu-eigen");
        Eigen::Tensor<float, 2> input(3, 3);
        input.setZero();
        auto output = relu.forward(input);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                ASSERT_NEAR(output(i, j), 0.0f, 1e-6f);
    }
    END_TEST("All zeros input produces all zeros output");

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
