/**
 * @file test_binary_cross_entropy.cpp
 * @brief Unit tests for the Binary Cross Entropy loss
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/losses/binary_cross_entropy.hpp"

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
    std::cout << "=== Binary Cross Entropy Loss Tests ===" << std::endl;

    // ---------------------------------------------------------------
    TEST("Perfect prediction gives near-zero loss")
    {
        CppNet::Losses::BinaryCrossEntropy bce("mean");
        Eigen::Tensor<float, 2> pred(1, 3);
        pred.setValues({{0.999f, 0.001f, 0.999f}});
        Eigen::Tensor<float, 2> target(1, 3);
        target.setValues({{1.0f, 0.0f, 1.0f}});

        float loss = bce.forward(pred, target);
        ASSERT_TRUE(loss < 0.01f);
    }
    END_TEST("Perfect prediction gives near-zero loss");

    // ---------------------------------------------------------------
    TEST("Worst prediction gives high loss")
    {
        CppNet::Losses::BinaryCrossEntropy bce("mean");
        Eigen::Tensor<float, 2> pred(1, 2);
        pred.setValues({{0.01f, 0.99f}});
        Eigen::Tensor<float, 2> target(1, 2);
        target.setValues({{1.0f, 0.0f}}); // opposite

        float loss = bce.forward(pred, target);
        ASSERT_TRUE(loss > 1.0f);
    }
    END_TEST("Worst prediction gives high loss");

    // ---------------------------------------------------------------
    TEST("Loss is always non-negative")
    {
        CppNet::Losses::BinaryCrossEntropy bce("mean");
        Eigen::Tensor<float, 2> pred(3, 4);
        pred.setRandom();
        // Clamp predictions to valid range (0, 1)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                pred(i, j) = std::max(0.01f, std::min(0.99f, std::fabs(pred(i, j))));

        Eigen::Tensor<float, 2> target(3, 4);
        target.setZero();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                target(i, j) = (j % 2 == 0) ? 1.0f : 0.0f;

        float loss = bce.forward(pred, target);
        ASSERT_TRUE(loss >= 0.0f);
    }
    END_TEST("Loss is always non-negative");

    // ---------------------------------------------------------------
    TEST("Known value: -[t*log(p) + (1-t)*log(1-p)]")
    {
        CppNet::Losses::BinaryCrossEntropy bce("mean");
        Eigen::Tensor<float, 2> pred(1, 1);
        pred.setValues({{0.5f}});
        Eigen::Tensor<float, 2> target(1, 1);
        target.setValues({{1.0f}});

        float loss = bce.forward(pred, target);
        // -[1*log(0.5) + 0*log(0.5)] = -log(0.5) ≈ 0.693
        ASSERT_NEAR(loss, -std::log(0.5f), 0.05f);
    }
    END_TEST("Known value: -[t*log(p) + (1-t)*log(1-p)]");

    // ---------------------------------------------------------------
    TEST("Backward - gradient shape matches")
    {
        CppNet::Losses::BinaryCrossEntropy bce("mean");
        Eigen::Tensor<float, 2> pred(2, 3);
        pred.setConstant(0.5f);
        Eigen::Tensor<float, 2> target(2, 3);
        target.setConstant(1.0f);

        auto grad = bce.backward(pred, target);
        ASSERT_TRUE(grad.dimension(0) == 2);
        ASSERT_TRUE(grad.dimension(1) == 3);
    }
    END_TEST("Backward - gradient shape matches");

    // ---------------------------------------------------------------
    TEST("Backward - gradient is finite")
    {
        CppNet::Losses::BinaryCrossEntropy bce("mean");
        Eigen::Tensor<float, 2> pred(2, 4);
        pred.setConstant(0.5f);
        Eigen::Tensor<float, 2> target(2, 4);
        target.setZero();
        target(0, 0) = 1.0f;
        target(1, 2) = 1.0f;

        auto grad = bce.backward(pred, target);
        for (int i = 0; i < grad.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad.data()[i]));
    }
    END_TEST("Backward - gradient is finite");

    // ---------------------------------------------------------------
    TEST("Backward - gradient direction for pred=0.5, target=1 → negative grad")
    {
        CppNet::Losses::BinaryCrossEntropy bce("mean");
        Eigen::Tensor<float, 2> pred(1, 1);
        pred.setValues({{0.5f}});
        Eigen::Tensor<float, 2> target(1, 1);
        target.setValues({{1.0f}});

        auto grad = bce.backward(pred, target);
        // dBCE/dpred = -(target/pred - (1-target)/(1-pred)) / N
        // For t=1, p=0.5: -(1/0.5) = -2/N → negative
        ASSERT_TRUE(grad(0, 0) < 0.0f);
    }
    END_TEST("Backward - gradient direction for pred=0.5, target=1 → negative grad");

    // ---------------------------------------------------------------
    TEST("Sum reduction vs mean reduction")
    {
        Eigen::Tensor<float, 2> pred(1, 4);
        pred.setConstant(0.7f);
        Eigen::Tensor<float, 2> target(1, 4);
        target.setConstant(1.0f);

        CppNet::Losses::BinaryCrossEntropy bce_mean("mean");
        CppNet::Losses::BinaryCrossEntropy bce_sum("sum");

        float loss_mean = bce_mean.forward(pred, target);
        float loss_sum = bce_sum.forward(pred, target);

        // Sum should be approximately 4x mean for 4 elements
        ASSERT_TRUE(loss_sum > loss_mean);
    }
    END_TEST("Sum reduction vs mean reduction");

    // ---------------------------------------------------------------
    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
