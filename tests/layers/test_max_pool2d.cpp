/**
 * @file test_max_pool2d.cpp
 * @brief Unit tests for the MaxPool2D layer
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/max_pool2d.hpp"

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
    std::cout << "=== MaxPool2D Layer Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Construction - default stride equals pool_size")
    {
        CppNet::Layers::MaxPool2D pool(2);
        ASSERT_TRUE(pool.get_pool_size() == 2);
        ASSERT_TRUE(pool.get_stride() == 2); // default: stride = pool_size
    }
    END_TEST("Construction - default stride equals pool_size");

    // ---------------------------------------------------------------
    TEST("Not trainable")
    {
        CppNet::Layers::MaxPool2D pool(2);
        ASSERT_TRUE(!pool.is_trainable());
    }
    END_TEST("Not trainable");

    // ---------------------------------------------------------------
    TEST("Forward - output shape [batch, ch, H/pool, W/pool]")
    {
        CppNet::Layers::MaxPool2D pool(2, -1, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 4, 4);
        input.setRandom();
        auto output = pool.forward(input);

        // 4/2 = 2
        ASSERT_TRUE(output.dimension(0) == 1);
        ASSERT_TRUE(output.dimension(1) == 1);
        ASSERT_TRUE(output.dimension(2) == 2);
        ASSERT_TRUE(output.dimension(3) == 2);
    }
    END_TEST("Forward - output shape [batch, ch, H/pool, W/pool]");

    // ---------------------------------------------------------------
    TEST("Forward - selects maximum values")
    {
        CppNet::Layers::MaxPool2D pool(2, -1, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 4, 4);
        // clang-format off
        input.setValues({{{{1.0f, 2.0f, 3.0f, 4.0f},
                           {5.0f, 6.0f, 7.0f, 8.0f},
                           {9.0f, 10.0f, 11.0f, 12.0f},
                           {13.0f, 14.0f, 15.0f, 16.0f}}}});
        // clang-format on
        auto output = pool.forward(input);

        // Pool 2x2: max of [1,2,5,6]=6, max of [3,4,7,8]=8
        //           max of [9,10,13,14]=14, max of [11,12,15,16]=16
        ASSERT_NEAR(output(0, 0, 0, 0), 6.0f, 1e-5f);
        ASSERT_NEAR(output(0, 0, 0, 1), 8.0f, 1e-5f);
        ASSERT_NEAR(output(0, 0, 1, 0), 14.0f, 1e-5f);
        ASSERT_NEAR(output(0, 0, 1, 1), 16.0f, 1e-5f);
    }
    END_TEST("Forward - selects maximum values");

    // ---------------------------------------------------------------
    TEST("Forward - multi-channel")
    {
        CppNet::Layers::MaxPool2D pool(2, -1, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 2, 4, 4);
        input.setRandom();
        auto output = pool.forward(input);

        ASSERT_TRUE(output.dimension(1) == 2); // channels preserved
        ASSERT_TRUE(output.dimension(2) == 2);
        ASSERT_TRUE(output.dimension(3) == 2);
    }
    END_TEST("Forward - multi-channel");

    // ---------------------------------------------------------------
    TEST("Forward - batch dimension preserved")
    {
        CppNet::Layers::MaxPool2D pool(2, -1, "cpu-eigen");
        Eigen::Tensor<float, 4> input(4, 3, 8, 8);
        input.setRandom();
        auto output = pool.forward(input);

        ASSERT_TRUE(output.dimension(0) == 4);
        ASSERT_TRUE(output.dimension(1) == 3);
        ASSERT_TRUE(output.dimension(2) == 4);
        ASSERT_TRUE(output.dimension(3) == 4);
    }
    END_TEST("Forward - batch dimension preserved");

    // ---------------------------------------------------------------
    TEST("Backward - grad input shape matches input")
    {
        CppNet::Layers::MaxPool2D pool(2, -1, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 4, 4);
        input.setRandom();
        auto output = pool.forward(input);

        Eigen::Tensor<float, 4> grad_output(output.dimensions());
        grad_output.setConstant(1.0f);
        auto grad_input = pool.backward(grad_output);

        ASSERT_TRUE(grad_input.dimension(0) == 1);
        ASSERT_TRUE(grad_input.dimension(1) == 1);
        ASSERT_TRUE(grad_input.dimension(2) == 4);
        ASSERT_TRUE(grad_input.dimension(3) == 4);
    }
    END_TEST("Backward - grad input shape matches input");

    // ---------------------------------------------------------------
    TEST("Forward - with stride != pool_size")
    {
        CppNet::Layers::MaxPool2D pool(3, 1, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 5, 5);
        input.setRandom();
        auto output = pool.forward(input);

        // Output H = (5 - 3)/1 + 1 = 3
        ASSERT_TRUE(output.dimension(2) == 3);
        ASSERT_TRUE(output.dimension(3) == 3);
    }
    END_TEST("Forward - with stride != pool_size");

    // ---------------------------------------------------------------
    TEST("Forward - all same values")
    {
        CppNet::Layers::MaxPool2D pool(2, -1, "cpu-eigen");
        Eigen::Tensor<float, 4> input(1, 1, 4, 4);
        input.setConstant(5.0f);
        auto output = pool.forward(input);

        for (int h = 0; h < output.dimension(2); ++h)
            for (int w = 0; w < output.dimension(3); ++w)
                ASSERT_NEAR(output(0, 0, h, w), 5.0f, 1e-5f);
    }
    END_TEST("Forward - all same values");

    // ---------------------------------------------------------------
    TEST("Output values are finite")
    {
        CppNet::Layers::MaxPool2D pool(2, -1, "cpu-eigen");
        Eigen::Tensor<float, 4> input(2, 3, 8, 8);
        input.setRandom();
        auto output = pool.forward(input);

        for (int b = 0; b < output.dimension(0); ++b)
            for (int c = 0; c < output.dimension(1); ++c)
                for (int h = 0; h < output.dimension(2); ++h)
                    for (int w = 0; w < output.dimension(3); ++w)
                        ASSERT_TRUE(std::isfinite(output(b, c, h, w)));
    }
    END_TEST("Output values are finite");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
