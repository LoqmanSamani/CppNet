/**
 * @file test_softmax.cpp
 * @brief Unit tests for the Softmax activation function
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/activations/softmax.hpp"

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
    std::cout << "=== Softmax Activation Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("2D forward - outputs sum to 1")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 2> input(2, 4);
        input.setValues({{1.0f, 2.0f, 3.0f, 4.0f}, {0.5f, 1.5f, -0.5f, 2.5f}});
        auto output = softmax.forward(input);

        for (int i = 0; i < 2; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 4; ++j)
                sum += output(i, j);
            ASSERT_NEAR(sum, 1.0f, 1e-5f);
        }
    }
    END_TEST("2D forward - outputs sum to 1");

    // ---------------------------------------------------------------
    TEST("2D forward - all outputs non-negative")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 2> input(2, 4);
        input.setValues({{-10.0f, -5.0f, 0.0f, 5.0f}, {-100.0f, 0.0f, 1.0f, 100.0f}});
        auto output = softmax.forward(input);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 4; ++j)
                ASSERT_TRUE(output(i, j) >= 0.0f);
    }
    END_TEST("2D forward - all outputs non-negative");

    // ---------------------------------------------------------------
    TEST("2D forward - equal inputs give uniform distribution")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 2> input(1, 5);
        input.setConstant(1.0f);
        auto output = softmax.forward(input);

        for (int j = 0; j < 5; ++j)
            ASSERT_NEAR(output(0, j), 0.2f, 1e-5f);
    }
    END_TEST("2D forward - equal inputs give uniform distribution");

    // ---------------------------------------------------------------
    TEST("2D forward - largest input gets largest probability")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 2> input(1, 4);
        input.setValues({{1.0f, 2.0f, 5.0f, 0.5f}});
        auto output = softmax.forward(input);

        // index 2 (value 5.0) should have the largest output
        ASSERT_TRUE(output(0, 2) > output(0, 0));
        ASSERT_TRUE(output(0, 2) > output(0, 1));
        ASSERT_TRUE(output(0, 2) > output(0, 3));
    }
    END_TEST("2D forward - largest input gets largest probability");

    // ---------------------------------------------------------------
    TEST("2D forward - single element is 1.0")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 2> input(1, 1);
        input(0, 0) = 42.0f;
        auto output = softmax.forward(input);
        ASSERT_NEAR(output(0, 0), 1.0f, 1e-5f);
    }
    END_TEST("2D forward - single element is 1.0");

    // ---------------------------------------------------------------
    TEST("2D forward - numerical stability with large values")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{1000.0f, 1001.0f, 1002.0f}});
        auto output = softmax.forward(input);

        float sum = 0.0f;
        for (int j = 0; j < 3; ++j) {
            ASSERT_TRUE(std::isfinite(output(0, j)));
            sum += output(0, j);
        }
        ASSERT_NEAR(sum, 1.0f, 1e-4f);
    }
    END_TEST("2D forward - numerical stability with large values");

    // ---------------------------------------------------------------
    TEST("2D backward - shape matches")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 2> input(3, 5);
        input.setRandom();
        softmax.forward(input);

        Eigen::Tensor<float, 2> grad(3, 5);
        grad.setConstant(1.0f);
        auto grad_input = softmax.backward(grad);

        ASSERT_TRUE(grad_input.dimension(0) == 3);
        ASSERT_TRUE(grad_input.dimension(1) == 5);
    }
    END_TEST("2D backward - shape matches");

    // ---------------------------------------------------------------
    TEST("2D backward - uniform grad gives zero gradient")
    {
        // When grad_output is uniform across features, softmax backward
        // should produce near-zero gradients (since all dL/dy_j are the same)
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 2> input(1, 4);
        input.setValues({{1.0f, 2.0f, 3.0f, 4.0f}});
        softmax.forward(input);

        Eigen::Tensor<float, 2> grad(1, 4);
        grad.setConstant(1.0f);
        auto grad_input = softmax.backward(grad);

        for (int j = 0; j < 4; ++j)
            ASSERT_NEAR(grad_input(0, j), 0.0f, 1e-5f);
    }
    END_TEST("2D backward - uniform grad gives zero gradient");

    // ---------------------------------------------------------------
    TEST("2D backward - gradient correctness via finite differences")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 2> input(1, 3);
        input.setValues({{1.0f, 2.0f, 3.0f}});
        auto output = softmax.forward(input);

        // Use one-hot grad for class 1
        Eigen::Tensor<float, 2> grad(1, 3);
        grad.setValues({{0.0f, 1.0f, 0.0f}});
        auto analytical_grad = softmax.backward(grad);

        // Numerical gradient via finite differences
        float eps = 1e-4f;
        for (int j = 0; j < 3; ++j) {
            Eigen::Tensor<float, 2> input_plus = input;
            input_plus(0, j) += eps;
            CppNet::Activations::Softmax sm_plus;
            auto out_plus = sm_plus.forward(input_plus);

            Eigen::Tensor<float, 2> input_minus = input;
            input_minus(0, j) -= eps;
            CppNet::Activations::Softmax sm_minus;
            auto out_minus = sm_minus.forward(input_minus);

            // numerical dL/dx_j = (L_plus - L_minus) / (2*eps) where L = output[0,1]
            float numerical = (out_plus(0, 1) - out_minus(0, 1)) / (2.0f * eps);
            ASSERT_NEAR(analytical_grad(0, j), numerical, 1e-3f);
        }
    }
    END_TEST("2D backward - gradient correctness via finite differences");

    // ---------------------------------------------------------------
    TEST("4D forward - outputs sum to 1 along last dim")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 4> input(1, 1, 2, 3);
        input.setRandom();
        auto output = softmax.forward(input);

        for (int a = 0; a < 1; ++a)
            for (int b = 0; b < 1; ++b)
                for (int c = 0; c < 2; ++c) {
                    float sum = 0.0f;
                    for (int d = 0; d < 3; ++d)
                        sum += output(a, b, c, d);
                    ASSERT_NEAR(sum, 1.0f, 1e-5f);
                }
    }
    END_TEST("4D forward - outputs sum to 1 along last dim");

    // ---------------------------------------------------------------
    TEST("4D backward - shape matches")
    {
        CppNet::Activations::Softmax softmax;
        Eigen::Tensor<float, 4> input(2, 3, 4, 5);
        input.setRandom();
        softmax.forward(input);

        Eigen::Tensor<float, 4> grad(2, 3, 4, 5);
        grad.setConstant(1.0f);
        auto grad_input = softmax.backward(grad);

        ASSERT_TRUE(grad_input.dimension(0) == 2);
        ASSERT_TRUE(grad_input.dimension(1) == 3);
        ASSERT_TRUE(grad_input.dimension(2) == 4);
        ASSERT_TRUE(grad_input.dimension(3) == 5);
    }
    END_TEST("4D backward - shape matches");

    // ---------------------------------------------------------------
    TEST("Invariance to constant shift")
    {
        CppNet::Activations::Softmax softmax1;
        Eigen::Tensor<float, 2> input(1, 4);
        input.setValues({{1.0f, 2.0f, 3.0f, 4.0f}});
        auto out1 = softmax1.forward(input);

        CppNet::Activations::Softmax softmax2;
        Eigen::Tensor<float, 2> input_shifted(1, 4);
        input_shifted.setValues({{101.0f, 102.0f, 103.0f, 104.0f}});
        auto out2 = softmax2.forward(input_shifted);

        for (int j = 0; j < 4; ++j)
            ASSERT_NEAR(out1(0, j), out2(0, j), 1e-4f);
    }
    END_TEST("Invariance to constant shift");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
