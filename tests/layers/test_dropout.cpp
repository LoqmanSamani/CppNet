/**
 * @file test_dropout.cpp
 * @brief Unit tests for the Dropout layer
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/dropout.hpp"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { std::cout << "  [TEST] " << name << " ... "; try {
#define END_TEST(name) std::cout << "PASSED\n"; ++tests_passed; \
    } catch (const std::exception& e) { std::cout << "FAILED: " << e.what() << "\n"; ++tests_failed; } \
      catch (...) { std::cout << "FAILED (unknown)\n"; ++tests_failed; } } while(0)
#define ASSERT_NEAR(a, b, eps) if (std::fabs((a)-(b))>(eps)) \
    throw std::runtime_error(std::string("Expected ")+std::to_string(b)+" got "+std::to_string(a))
#define ASSERT_TRUE(c) if (!(c)) throw std::runtime_error("Assertion failed: " #c)

int main() {
    std::cout << "=== Dropout Tests ===\n";

    TEST("Construction") {
        CppNet::Layers::Dropout dropout(0.3f);
        ASSERT_NEAR(dropout.get_p(), 0.3f, 1e-6f);
        ASSERT_TRUE(!dropout.is_trainable());
    } END_TEST("Construction");

    TEST("Default p = 0.5") {
        CppNet::Layers::Dropout dropout;
        ASSERT_NEAR(dropout.get_p(), 0.5f, 1e-6f);
    } END_TEST("Default p = 0.5");

    TEST("Training: some elements zeroed (2D)") {
        CppNet::Layers::Dropout dropout(0.5f);
        dropout.train();
        Eigen::Tensor<float, 2> input(4, 100);
        input.setConstant(1.0f);
        auto out = dropout.forward(input);
        int zeros = 0;
        for (int i = 0; i < out.size(); ++i)
            if (out.data()[i] == 0.0f) ++zeros;
        // With p=0.5, roughly half should be zero (allow wide margin)
        ASSERT_TRUE(zeros > 50);
        ASSERT_TRUE(zeros < 350);
    } END_TEST("Training: some elements zeroed (2D)");

    TEST("Training: inverted scaling") {
        CppNet::Layers::Dropout dropout(0.5f);
        dropout.train();
        Eigen::Tensor<float, 2> input(1, 1000);
        input.setConstant(1.0f);
        auto out = dropout.forward(input);
        // Non-zero elements should be scaled by 1/(1-p) = 2
        for (int i = 0; i < out.size(); ++i)
            if (out.data()[i] != 0.0f)
                ASSERT_NEAR(out.data()[i], 2.0f, 1e-5f);
    } END_TEST("Training: inverted scaling");

    TEST("Eval mode: identity") {
        CppNet::Layers::Dropout dropout(0.5f);
        dropout.eval();
        Eigen::Tensor<float, 2> input(2, 4);
        input.setRandom();
        auto out = dropout.forward(input);
        for (int i = 0; i < out.size(); ++i)
            ASSERT_NEAR(out.data()[i], input.data()[i], 1e-6f);
    } END_TEST("Eval mode: identity");

    TEST("4D forward shape") {
        CppNet::Layers::Dropout dropout(0.3f);
        dropout.train();
        Eigen::Tensor<float, 4> input(2, 3, 4, 4);
        input.setConstant(1.0f);
        auto out = dropout.forward(input);
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(1) == 3);
        ASSERT_TRUE(out.dimension(2) == 4);
        ASSERT_TRUE(out.dimension(3) == 4);
    } END_TEST("4D forward shape");

    TEST("Backward shape (2D)") {
        CppNet::Layers::Dropout dropout(0.5f);
        dropout.train();
        Eigen::Tensor<float, 2> input(3, 5);
        input.setConstant(1.0f);
        dropout.forward(input);
        Eigen::Tensor<float, 2> grad(3, 5);
        grad.setConstant(1.0f);
        auto grad_in = dropout.backward(grad);
        ASSERT_TRUE(grad_in.dimension(0) == 3);
        ASSERT_TRUE(grad_in.dimension(1) == 5);
    } END_TEST("Backward shape (2D)");

    TEST("Backward uses same mask as forward") {
        CppNet::Layers::Dropout dropout(0.5f);
        dropout.train();
        Eigen::Tensor<float, 2> input(1, 100);
        input.setConstant(1.0f);
        auto out = dropout.forward(input);
        Eigen::Tensor<float, 2> grad(1, 100);
        grad.setConstant(1.0f);
        auto grad_in = dropout.backward(grad);
        // Where out was 0, grad_in should also be 0
        for (int i = 0; i < 100; ++i) {
            if (out(0, i) == 0.0f)
                ASSERT_NEAR(grad_in(0, i), 0.0f, 1e-6f);
        }
    } END_TEST("Backward uses same mask as forward");

    TEST("p=0 means no dropout") {
        CppNet::Layers::Dropout dropout(0.0f);
        dropout.train();
        Eigen::Tensor<float, 2> input(2, 5);
        input.setRandom();
        auto out = dropout.forward(input);
        for (int i = 0; i < out.size(); ++i)
            ASSERT_NEAR(out.data()[i], input.data()[i], 1e-5f);
    } END_TEST("p=0 means no dropout");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
