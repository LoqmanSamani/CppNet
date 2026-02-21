/**
 * @file test_leacky_relu.cpp
 * @brief Unit tests for the LeakyReLU activation function
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/activations/leacky_relu.hpp"

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
    std::cout << "=== LeakyReLU Activation Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("2D forward - positive values pass through")
    {
        CppNet::Activations::LeakyReLU lrelu(0.01f, "cpu-eigen");
        Eigen::Tensor<float, 2> input(2, 3);
        input.setValues({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
        auto output = lrelu.forward(input);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                ASSERT_NEAR(output(i, j), input(i, j), 1e-6f);
    }
    END_TEST("2D forward - positive values pass through");

    // ---------------------------------------------------------------
    TEST("2D forward - negative values scaled by alpha")
    {
        float alpha = 0.01f;
        CppNet::Activations::LeakyReLU lrelu(alpha, "cpu-eigen");
        Eigen::Tensor<float, 2> input(1, 4);
        input.setValues({{-1.0f, -2.0f, -0.5f, -100.0f}});
        auto output = lrelu.forward(input);

        for (int j = 0; j < 4; ++j)
            ASSERT_NEAR(output(0, j), alpha * input(0, j), 1e-5f);
    }
    END_TEST("2D forward - negative values scaled by alpha");

    // ---------------------------------------------------------------
    TEST("2D forward - mixed values")
    {
        float alpha = 0.1f;
        CppNet::Activations::LeakyReLU lrelu(alpha, "cpu-eigen");
        Eigen::Tensor<float, 2> input(1, 4);
        input.setValues({{-2.0f, 3.0f, -0.5f, 1.0f}});
        auto output = lrelu.forward(input);

        ASSERT_NEAR(output(0, 0), alpha * (-2.0f), 1e-5f);
        ASSERT_NEAR(output(0, 1), 3.0f, 1e-5f);
        ASSERT_NEAR(output(0, 2), alpha * (-0.5f), 1e-5f);
        ASSERT_NEAR(output(0, 3), 1.0f, 1e-5f);
    }
    END_TEST("2D forward - mixed values");

    // ---------------------------------------------------------------
    TEST("2D forward - different alpha values")
    {
        float alpha = 0.2f;
        CppNet::Activations::LeakyReLU lrelu(alpha, "cpu-eigen");
        Eigen::Tensor<float, 2> input(1, 2);
        input.setValues({{-5.0f, 5.0f}});
        auto output = lrelu.forward(input);

        ASSERT_NEAR(output(0, 0), -1.0f, 1e-5f); // 0.2 * -5 = -1
        ASSERT_NEAR(output(0, 1), 5.0f, 1e-5f);
    }
    END_TEST("2D forward - different alpha values");

    // ---------------------------------------------------------------
    TEST("get_alpha returns correct value")
    {
        CppNet::Activations::LeakyReLU lrelu(0.05f, "cpu-eigen");
        ASSERT_NEAR(lrelu.get_alpha(), 0.05f, 1e-6f);
    }
    END_TEST("get_alpha returns correct value");

    // ---------------------------------------------------------------
    TEST("2D backward - positive region: grad passes through")
    {
        CppNet::Activations::LeakyReLU lrelu(0.01f, "cpu-eigen");
        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{1.0f, 2.0f, 3.0f}});
        lrelu.forward(input);

        Eigen::Tensor<float, 2> grad(1, 3);
        grad.setConstant(1.0f);
        auto grad_input = lrelu.backward(grad);

        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(grad_input(0, j), 1.0f, 1e-6f);
    }
    END_TEST("2D backward - positive region: grad passes through");

    // ---------------------------------------------------------------
    TEST("2D backward - negative region: grad scaled by alpha")
    {
        float alpha = 0.01f;
        CppNet::Activations::LeakyReLU lrelu(alpha, "cpu-eigen");
        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{-1.0f, -2.0f, -3.0f}});
        lrelu.forward(input);

        Eigen::Tensor<float, 2> grad(1, 3);
        grad.setConstant(1.0f);
        auto grad_input = lrelu.backward(grad);

        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(grad_input(0, j), alpha, 1e-5f);
    }
    END_TEST("2D backward - negative region: grad scaled by alpha");

    // ---------------------------------------------------------------
    TEST("2D backward - mixed region gradient")
    {
        float alpha = 0.1f;
        CppNet::Activations::LeakyReLU lrelu(alpha, "cpu-eigen");
        Eigen::Tensor<float, 2> input(1, 4);
        input.setValues({{-2.0f, 3.0f, -0.5f, 1.0f}});
        lrelu.forward(input);

        Eigen::Tensor<float, 2> grad(1, 4);
        grad.setValues({{2.0f, 3.0f, 4.0f, 5.0f}});
        auto grad_input = lrelu.backward(grad);

        ASSERT_NEAR(grad_input(0, 0), alpha * 2.0f, 1e-5f);
        ASSERT_NEAR(grad_input(0, 1), 3.0f, 1e-5f);
        ASSERT_NEAR(grad_input(0, 2), alpha * 4.0f, 1e-5f);
        ASSERT_NEAR(grad_input(0, 3), 5.0f, 1e-5f);
    }
    END_TEST("2D backward - mixed region gradient");

    // ---------------------------------------------------------------
    TEST("4D forward - basic operation")
    {
        float alpha = 0.01f;
        CppNet::Activations::LeakyReLU lrelu(alpha, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 2, 2);
        input.setValues({{{{-1.0f, 2.0f}, {3.0f, -4.0f}}}});
        auto output = lrelu.forward(input);

        ASSERT_NEAR(output(0, 0, 0, 0), alpha * (-1.0f), 1e-5f);
        ASSERT_NEAR(output(0, 0, 0, 1), 2.0f, 1e-5f);
        ASSERT_NEAR(output(0, 0, 1, 0), 3.0f, 1e-5f);
        ASSERT_NEAR(output(0, 0, 1, 1), alpha * (-4.0f), 1e-5f);
    }
    END_TEST("4D forward - basic operation");

    // ---------------------------------------------------------------
    TEST("4D backward - shape matches")
    {
        CppNet::Activations::LeakyReLU lrelu(0.01f, "cpu-eigen");
        Eigen::Tensor<float, 4> input(2, 3, 4, 4);
        input.setRandom();
        lrelu.forward(input);

        Eigen::Tensor<float, 4> grad(2, 3, 4, 4);
        grad.setConstant(1.0f);
        auto grad_input = lrelu.backward(grad);

        ASSERT_TRUE(grad_input.dimension(0) == 2);
        ASSERT_TRUE(grad_input.dimension(1) == 3);
        ASSERT_TRUE(grad_input.dimension(2) == 4);
        ASSERT_TRUE(grad_input.dimension(3) == 4);
    }
    END_TEST("4D backward - shape matches");

    // ---------------------------------------------------------------
    TEST("Default alpha is 0.01")
    {
        CppNet::Activations::LeakyReLU lrelu;
        ASSERT_NEAR(lrelu.get_alpha(), 0.01f, 1e-6f);
    }
    END_TEST("Default alpha is 0.01");

    // ---------------------------------------------------------------
    TEST("alpha=0 degenerates to ReLU")
    {
        CppNet::Activations::LeakyReLU lrelu(0.0f, "cpu-eigen");
        Eigen::Tensor<float, 2> input(1, 4);
        input.setValues({{-2.0f, -1.0f, 1.0f, 2.0f}});
        auto output = lrelu.forward(input);

        ASSERT_NEAR(output(0, 0), 0.0f, 1e-6f);
        ASSERT_NEAR(output(0, 1), 0.0f, 1e-6f);
        ASSERT_NEAR(output(0, 2), 1.0f, 1e-6f);
        ASSERT_NEAR(output(0, 3), 2.0f, 1e-6f);
    }
    END_TEST("alpha=0 degenerates to ReLU");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
