/**
 * @file test_tanh.cpp
 * @brief Unit tests for the Tanh activation function
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/activations/tanh.hpp"

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
    std::cout << "=== Tanh Activation Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("2D forward - tanh(0) = 0")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 2> input(1, 1);
        input(0, 0) = 0.0f;
        auto output = tanh_act.forward(input);
        ASSERT_NEAR(output(0, 0), 0.0f, 1e-6f);
    }
    END_TEST("2D forward - tanh(0) = 0");

    // ---------------------------------------------------------------
    TEST("2D forward - output in (-1, 1)")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 2> input(2, 3);
        input.setValues({{-10.0f, 0.0f, 10.0f}, {-100.0f, 0.5f, 100.0f}});
        auto output = tanh_act.forward(input);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j) {
                ASSERT_TRUE(output(i, j) >= -1.0f);
                ASSERT_TRUE(output(i, j) <= 1.0f);
            }
    }
    END_TEST("2D forward - output in (-1, 1)");

    // ---------------------------------------------------------------
    TEST("2D forward - known values")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 2> input(1, 4);
        input.setValues({{-2.0f, -1.0f, 1.0f, 2.0f}});
        auto output = tanh_act.forward(input);

        for (int j = 0; j < 4; ++j)
            ASSERT_NEAR(output(0, j), std::tanh(input(0, j)), 1e-5f);
    }
    END_TEST("2D forward - known values");

    // ---------------------------------------------------------------
    TEST("2D forward - odd function: tanh(-x) = -tanh(x)")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{1.0f, 2.0f, 3.0f}});
        auto out_pos = tanh_act.forward(input);

        CppNet::Activations::Tanh tanh_act2;
        Eigen::Tensor<float, 2> input_neg(1, 3);
        input_neg.setValues({{-1.0f, -2.0f, -3.0f}});
        auto out_neg = tanh_act2.forward(input_neg);

        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(out_neg(0, j), -out_pos(0, j), 1e-5f);
    }
    END_TEST("2D forward - odd function: tanh(-x) = -tanh(x)");

    // ---------------------------------------------------------------
    TEST("2D backward - gradient = 1 - tanh^2")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{-1.0f, 0.0f, 1.0f}});
        auto output = tanh_act.forward(input);

        Eigen::Tensor<float, 2> grad(1, 3);
        grad.setConstant(1.0f);
        auto grad_input = tanh_act.backward(grad);

        for (int j = 0; j < 3; ++j) {
            float t = output(0, j);
            float expected = 1.0f - t * t;
            ASSERT_NEAR(grad_input(0, j), expected, 1e-5f);
        }
    }
    END_TEST("2D backward - gradient = 1 - tanh^2");

    // ---------------------------------------------------------------
    TEST("2D backward - max gradient at x=0")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{-3.0f, 0.0f, 3.0f}});
        tanh_act.forward(input);

        Eigen::Tensor<float, 2> grad(1, 3);
        grad.setConstant(1.0f);
        auto grad_input = tanh_act.backward(grad);

        ASSERT_NEAR(grad_input(0, 1), 1.0f, 1e-5f); // tanh'(0) = 1
        ASSERT_TRUE(grad_input(0, 1) > grad_input(0, 0));
        ASSERT_TRUE(grad_input(0, 1) > grad_input(0, 2));
    }
    END_TEST("2D backward - max gradient at x=0");

    // ---------------------------------------------------------------
    TEST("2D backward - shape matches")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 2> input(4, 5);
        input.setRandom();
        tanh_act.forward(input);

        Eigen::Tensor<float, 2> grad(4, 5);
        grad.setConstant(1.0f);
        auto grad_input = tanh_act.backward(grad);

        ASSERT_TRUE(grad_input.dimension(0) == 4);
        ASSERT_TRUE(grad_input.dimension(1) == 5);
    }
    END_TEST("2D backward - shape matches");

    // ---------------------------------------------------------------
    TEST("4D forward - basic operation")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 4> input(1, 1, 2, 2);
        input.setValues({{{{-1.0f, 0.0f}, {0.0f, 1.0f}}}});
        auto output = tanh_act.forward(input);

        ASSERT_NEAR(output(0, 0, 0, 0), std::tanh(-1.0f), 1e-5f);
        ASSERT_NEAR(output(0, 0, 0, 1), 0.0f, 1e-5f);
        ASSERT_NEAR(output(0, 0, 1, 0), 0.0f, 1e-5f);
        ASSERT_NEAR(output(0, 0, 1, 1), std::tanh(1.0f), 1e-5f);
    }
    END_TEST("4D forward - basic operation");

    // ---------------------------------------------------------------
    TEST("4D backward - shape matches")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 4> input(2, 3, 4, 4);
        input.setRandom();
        tanh_act.forward(input);

        Eigen::Tensor<float, 4> grad(2, 3, 4, 4);
        grad.setConstant(1.0f);
        auto grad_input = tanh_act.backward(grad);

        ASSERT_TRUE(grad_input.dimension(0) == 2);
        ASSERT_TRUE(grad_input.dimension(1) == 3);
        ASSERT_TRUE(grad_input.dimension(2) == 4);
        ASSERT_TRUE(grad_input.dimension(3) == 4);
    }
    END_TEST("4D backward - shape matches");

    // ---------------------------------------------------------------
    TEST("Large input saturation")
    {
        CppNet::Activations::Tanh tanh_act;
        Eigen::Tensor<float, 2> input(1, 2);
        input.setValues({{50.0f, -50.0f}});
        auto output = tanh_act.forward(input);
        ASSERT_NEAR(output(0, 0), 1.0f, 1e-5f);
        ASSERT_NEAR(output(0, 1), -1.0f, 1e-5f);
    }
    END_TEST("Large input saturation");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
