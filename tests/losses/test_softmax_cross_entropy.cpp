/**
 * @file test_softmax_cross_entropy.cpp
 * @brief Unit tests for the SoftmaxCrossEntropy loss
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/losses/softmax_cross_entropy.hpp"

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
    std::cout << "=== SoftmaxCrossEntropy Tests ===\n";

    TEST("Construction default") {
        CppNet::Losses::SoftmaxCrossEntropy loss;
        // Should not throw
        ASSERT_TRUE(true);
    } END_TEST("Construction default");

    TEST("Perfect prediction gives low loss") {
        CppNet::Losses::SoftmaxCrossEntropy loss("mean");
        Eigen::Tensor<float, 2> logits(1, 3);
        logits.setValues({{10.0f, -10.0f, -10.0f}});
        Eigen::Tensor<float, 2> targets(1, 3);
        targets.setValues({{1.0f, 0.0f, 0.0f}});
        float l = loss.forward(logits, targets);
        ASSERT_TRUE(l < 0.1f);
    } END_TEST("Perfect prediction gives low loss");

    TEST("Wrong prediction gives high loss") {
        CppNet::Losses::SoftmaxCrossEntropy loss("mean");
        Eigen::Tensor<float, 2> logits(1, 3);
        logits.setValues({{-10.0f, 10.0f, -10.0f}});
        Eigen::Tensor<float, 2> targets(1, 3);
        targets.setValues({{1.0f, 0.0f, 0.0f}});
        float l = loss.forward(logits, targets);
        ASSERT_TRUE(l > 10.0f);
    } END_TEST("Wrong prediction gives high loss");

    TEST("Loss is non-negative") {
        CppNet::Losses::SoftmaxCrossEntropy loss("mean");
        Eigen::Tensor<float, 2> logits(4, 5);
        logits.setRandom();
        Eigen::Tensor<float, 2> targets(4, 5);
        targets.setZero();
        for (int i = 0; i < 4; ++i) targets(i, i % 5) = 1.0f;
        float l = loss.forward(logits, targets);
        ASSERT_TRUE(l >= 0.0f);
    } END_TEST("Loss is non-negative");

    TEST("Backward shape") {
        CppNet::Losses::SoftmaxCrossEntropy loss("mean");
        Eigen::Tensor<float, 2> logits(2, 3);
        logits.setRandom();
        Eigen::Tensor<float, 2> targets(2, 3);
        targets.setZero();
        targets(0, 0) = 1.0f;
        targets(1, 2) = 1.0f;
        loss.forward(logits, targets);
        auto grad = loss.backward(logits, targets);
        ASSERT_TRUE(grad.dimension(0) == 2);
        ASSERT_TRUE(grad.dimension(1) == 3);
    } END_TEST("Backward shape");

    TEST("Backward: grad = softmax - target") {
        CppNet::Losses::SoftmaxCrossEntropy loss("mean");
        Eigen::Tensor<float, 2> logits(1, 3);
        logits.setValues({{1.0f, 2.0f, 3.0f}});
        Eigen::Tensor<float, 2> targets(1, 3);
        targets.setValues({{0.0f, 0.0f, 1.0f}});
        loss.forward(logits, targets);
        auto grad = loss.backward(logits, targets);
        // softmax(1,2,3) ≈ (0.0900, 0.2447, 0.6652)
        // grad ≈ softmax - target
        ASSERT_NEAR(grad(0, 0), 0.0900f, 0.02f);
        ASSERT_NEAR(grad(0, 1), 0.2447f, 0.02f);
        ASSERT_NEAR(grad(0, 2), -0.3348f, 0.02f);
    } END_TEST("Backward: grad = softmax - target");

    TEST("Backward is finite") {
        CppNet::Losses::SoftmaxCrossEntropy loss("mean");
        Eigen::Tensor<float, 2> logits(4, 5);
        logits.setRandom();
        Eigen::Tensor<float, 2> targets(4, 5);
        targets.setZero();
        for (int i = 0; i < 4; ++i) targets(i, i % 5) = 1.0f;
        loss.forward(logits, targets);
        auto grad = loss.backward(logits, targets);
        for (int i = 0; i < grad.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad.data()[i]));
    } END_TEST("Backward is finite");

    TEST("Sum reduction gives larger value") {
        CppNet::Losses::SoftmaxCrossEntropy loss_mean("mean");
        CppNet::Losses::SoftmaxCrossEntropy loss_sum("sum");
        Eigen::Tensor<float, 2> logits(4, 3);
        logits.setRandom();
        Eigen::Tensor<float, 2> targets(4, 3);
        targets.setZero();
        for (int i = 0; i < 4; ++i) targets(i, i % 3) = 1.0f;
        float lm = loss_mean.forward(logits, targets);
        float ls = loss_sum.forward(logits, targets);
        ASSERT_TRUE(ls >= lm - 1e-5f);
    } END_TEST("Sum reduction gives larger value");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
