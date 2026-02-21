/**
 * @file test_sigmoid.cpp
 * @brief Unit tests for the Sigmoid activation function
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/activations/sigmoid.hpp"

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

static float sigmoid_ref(float x) {
    float clamped = std::max(-500.0f, std::min(500.0f, x));
    return 1.0f / (1.0f + std::exp(-clamped));
}

int main()
{
    std::cout << "=== Sigmoid Activation Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("2D forward - output in (0, 1)")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 2> input(2, 3);
        input.setValues({{-5.0f, 0.0f, 5.0f}, {-100.0f, 1.0f, 100.0f}});
        auto output = sigmoid.forward(input);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j) {
                ASSERT_TRUE(output(i, j) >= 0.0f);
                ASSERT_TRUE(output(i, j) <= 1.0f);
            }
    }
    END_TEST("2D forward - output in (0, 1)");

    // ---------------------------------------------------------------
    TEST("2D forward - sigmoid(0) = 0.5")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 2> input(1, 1);
        input(0, 0) = 0.0f;
        auto output = sigmoid.forward(input);
        ASSERT_NEAR(output(0, 0), 0.5f, 1e-5f);
    }
    END_TEST("2D forward - sigmoid(0) = 0.5");

    // ---------------------------------------------------------------
    TEST("2D forward - known values")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 2> input(1, 4);
        input.setValues({{-2.0f, -1.0f, 1.0f, 2.0f}});
        auto output = sigmoid.forward(input);

        for (int j = 0; j < 4; ++j)
            ASSERT_NEAR(output(0, j), sigmoid_ref(input(0, j)), 1e-5f);
    }
    END_TEST("2D forward - known values");

    // ---------------------------------------------------------------
    TEST("2D forward - large positive values near 1")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 2> input(1, 1);
        input(0, 0) = 50.0f;
        auto output = sigmoid.forward(input);
        ASSERT_NEAR(output(0, 0), 1.0f, 1e-5f);
    }
    END_TEST("2D forward - large positive values near 1");

    // ---------------------------------------------------------------
    TEST("2D forward - large negative values near 0")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 2> input(1, 1);
        input(0, 0) = -50.0f;
        auto output = sigmoid.forward(input);
        ASSERT_NEAR(output(0, 0), 0.0f, 1e-5f);
    }
    END_TEST("2D forward - large negative values near 0");

    // ---------------------------------------------------------------
    TEST("2D backward - gradient shape matches")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 2> input(3, 4);
        input.setRandom();
        sigmoid.forward(input);

        Eigen::Tensor<float, 2> grad(3, 4);
        grad.setConstant(1.0f);
        auto grad_input = sigmoid.backward(grad);

        ASSERT_TRUE(grad_input.dimension(0) == 3);
        ASSERT_TRUE(grad_input.dimension(1) == 4);
    }
    END_TEST("2D backward - gradient shape matches");

    // ---------------------------------------------------------------
    TEST("2D backward - gradient = sigmoid * (1 - sigmoid)")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{-1.0f, 0.0f, 1.0f}});
        auto output = sigmoid.forward(input);

        Eigen::Tensor<float, 2> grad(1, 3);
        grad.setConstant(1.0f);
        auto grad_input = sigmoid.backward(grad);

        for (int j = 0; j < 3; ++j) {
            float s = output(0, j);
            float expected = s * (1.0f - s);
            ASSERT_NEAR(grad_input(0, j), expected, 1e-5f);
        }
    }
    END_TEST("2D backward - gradient = sigmoid * (1 - sigmoid)");

    // ---------------------------------------------------------------
    TEST("2D backward - max gradient at x=0")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{-5.0f, 0.0f, 5.0f}});
        sigmoid.forward(input);

        Eigen::Tensor<float, 2> grad(1, 3);
        grad.setConstant(1.0f);
        auto grad_input = sigmoid.backward(grad);

        // Gradient at x=0 should be ~0.25, and larger than at x=-5 or x=5
        ASSERT_NEAR(grad_input(0, 1), 0.25f, 1e-4f);
        ASSERT_TRUE(grad_input(0, 1) > grad_input(0, 0));
        ASSERT_TRUE(grad_input(0, 1) > grad_input(0, 2));
    }
    END_TEST("2D backward - max gradient at x=0");

    // ---------------------------------------------------------------
    TEST("4D forward - basic operation")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 4> input(1, 1, 2, 2);
        input.setValues({{{{-2.0f, 0.0f}, {0.0f, 2.0f}}}});
        auto output = sigmoid.forward(input);

        ASSERT_NEAR(output(0, 0, 0, 1), 0.5f, 1e-5f);
        ASSERT_NEAR(output(0, 0, 1, 0), 0.5f, 1e-5f);
        ASSERT_TRUE(output(0, 0, 0, 0) < 0.5f);
        ASSERT_TRUE(output(0, 0, 1, 1) > 0.5f);
    }
    END_TEST("4D forward - basic operation");

    // ---------------------------------------------------------------
    TEST("4D backward - shape matches")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 4> input(2, 3, 4, 4);
        input.setRandom();
        sigmoid.forward(input);

        Eigen::Tensor<float, 4> grad(2, 3, 4, 4);
        grad.setConstant(1.0f);
        auto grad_input = sigmoid.backward(grad);

        ASSERT_TRUE(grad_input.dimension(0) == 2);
        ASSERT_TRUE(grad_input.dimension(1) == 3);
        ASSERT_TRUE(grad_input.dimension(2) == 4);
        ASSERT_TRUE(grad_input.dimension(3) == 4);
    }
    END_TEST("4D backward - shape matches");

    // ---------------------------------------------------------------
    TEST("Symmetry: sigmoid(-x) = 1 - sigmoid(x)")
    {
        CppNet::Activations::Sigmoid sigmoid;
        Eigen::Tensor<float, 2> input_pos(1, 3);
        input_pos.setValues({{1.0f, 2.0f, 3.0f}});
        auto out_pos = sigmoid.forward(input_pos);

        CppNet::Activations::Sigmoid sigmoid2;
        Eigen::Tensor<float, 2> input_neg(1, 3);
        input_neg.setValues({{-1.0f, -2.0f, -3.0f}});
        auto out_neg = sigmoid2.forward(input_neg);

        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(out_neg(0, j), 1.0f - out_pos(0, j), 1e-5f);
    }
    END_TEST("Symmetry: sigmoid(-x) = 1 - sigmoid(x)");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
