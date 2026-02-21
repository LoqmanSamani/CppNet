/**
 * @file test_categorical_cross_entropy.cpp
 * @brief Unit tests for Categorical Cross Entropy loss
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/losses/categorical_cross_entropy.hpp"

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
    std::cout << "=== Categorical Cross Entropy Loss Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("One-hot: perfect prediction gives near-zero loss")
    {
        CppNet::Losses::CategoricalCrossEntropy cce("mean", true);
        Eigen::Tensor<float, 2> pred(2, 3);
        // Large logits for correct class
        pred.setValues({{10.0f, -10.0f, -10.0f},
                        {-10.0f, -10.0f, 10.0f}});
        Eigen::Tensor<float, 2> target(2, 3);
        target.setValues({{1.0f, 0.0f, 0.0f},
                          {0.0f, 0.0f, 1.0f}});

        float loss = cce.forward(pred, target);
        ASSERT_TRUE(loss < 0.1f);
    }
    END_TEST("One-hot: perfect prediction gives near-zero loss");

    // ---------------------------------------------------------------
    TEST("One-hot: wrong prediction gives high loss")
    {
        CppNet::Losses::CategoricalCrossEntropy cce("mean", true);
        Eigen::Tensor<float, 2> pred(1, 3);
        // All equal logits - uniform distribution
        pred.setValues({{0.0f, 0.0f, 0.0f}});
        Eigen::Tensor<float, 2> target(1, 3);
        target.setValues({{1.0f, 0.0f, 0.0f}});

        float loss = cce.forward(pred, target);
        // -log(1/3) ≈ 1.0986
        ASSERT_TRUE(loss > 0.5f);
    }
    END_TEST("One-hot: wrong prediction gives high loss");

    // ---------------------------------------------------------------
    TEST("One-hot: loss is non-negative")
    {
        CppNet::Losses::CategoricalCrossEntropy cce("mean", true);
        Eigen::Tensor<float, 2> pred(4, 5);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(4, 5);
        target.setZero();
        for (int i = 0; i < 4; ++i)
            target(i, i % 5) = 1.0f;

        float loss = cce.forward(pred, target);
        ASSERT_TRUE(loss >= 0.0f);
    }
    END_TEST("One-hot: loss is non-negative");

    // ---------------------------------------------------------------
    TEST("Index targets: basic forward")
    {
        CppNet::Losses::CategoricalCrossEntropy cce("mean", true);
        Eigen::Tensor<float, 2> pred(3, 4);
        pred.setRandom();
        Eigen::Tensor<int, 1> target(3);
        target.setValues({0, 2, 3});

        float loss = cce.forward(pred, target);
        ASSERT_TRUE(loss >= 0.0f);
        ASSERT_TRUE(std::isfinite(loss));
    }
    END_TEST("Index targets: basic forward");

    // ---------------------------------------------------------------
    TEST("One-hot backward: gradient shape matches")
    {
        CppNet::Losses::CategoricalCrossEntropy cce("mean", true);
        Eigen::Tensor<float, 2> pred(2, 4);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(2, 4);
        target.setZero();
        target(0, 1) = 1.0f;
        target(1, 3) = 1.0f;

        cce.forward(pred, target);
        auto grad = cce.backward(pred, target);

        ASSERT_TRUE(grad.dimension(0) == 2);
        ASSERT_TRUE(grad.dimension(1) == 4);
    }
    END_TEST("One-hot backward: gradient shape matches");

    // ---------------------------------------------------------------
    TEST("One-hot backward: gradient is finite")
    {
        CppNet::Losses::CategoricalCrossEntropy cce("mean", true);
        Eigen::Tensor<float, 2> pred(2, 3);
        pred.setRandom();
        Eigen::Tensor<float, 2> target(2, 3);
        target.setZero();
        target(0, 0) = 1.0f;
        target(1, 2) = 1.0f;

        cce.forward(pred, target);
        auto grad = cce.backward(pred, target);

        for (int i = 0; i < grad.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad.data()[i]));
    }
    END_TEST("One-hot backward: gradient is finite");

    // ---------------------------------------------------------------
    TEST("Index backward: gradient shape matches")
    {
        CppNet::Losses::CategoricalCrossEntropy cce("mean", true);
        Eigen::Tensor<float, 2> pred(3, 5);
        pred.setRandom();
        Eigen::Tensor<int, 1> target(3);
        target.setValues({0, 2, 4});

        cce.forward(pred, target);
        auto grad = cce.backward(pred, target);

        ASSERT_TRUE(grad.dimension(0) == 3);
        ASSERT_TRUE(grad.dimension(1) == 5);
    }
    END_TEST("Index backward: gradient shape matches");

    // ---------------------------------------------------------------
    TEST("Index backward: gradient is finite")
    {
        CppNet::Losses::CategoricalCrossEntropy cce("mean", true);
        Eigen::Tensor<float, 2> pred(2, 4);
        pred.setRandom();
        Eigen::Tensor<int, 1> target(2);
        target.setValues({1, 3});

        cce.forward(pred, target);
        auto grad = cce.backward(pred, target);

        for (int i = 0; i < grad.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad.data()[i]));
    }
    END_TEST("Index backward: gradient is finite");

    // ---------------------------------------------------------------
    TEST("Loss is finite with random data")
    {
        CppNet::Losses::CategoricalCrossEntropy cce("mean", true);
        Eigen::Tensor<float, 2> pred(8, 10);
        pred.setRandom();
        Eigen::Tensor<int, 1> target(8);
        for (int i = 0; i < 8; ++i)
            target(i) = i % 10;

        float loss = cce.forward(pred, target);
        ASSERT_TRUE(std::isfinite(loss));
    }
    END_TEST("Loss is finite with random data");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
