/**
 * @file test_batch_norm.cpp
 * @brief Unit tests for the BatchNorm layer
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/batch_norm.hpp"

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
    std::cout << "=== BatchNorm Tests ===\n";

    TEST("Construction") {
        CppNet::Layers::BatchNorm bn(8);
        ASSERT_TRUE(bn.get_num_features() == 8);
        ASSERT_TRUE(bn.is_trainable());
    } END_TEST("Construction");

    TEST("Gamma initialized to ones, beta to zeros") {
        CppNet::Layers::BatchNorm bn(4);
        auto& g = bn.get_gamma();
        auto& b = bn.get_beta();
        for (int i = 0; i < 4; ++i) {
            ASSERT_NEAR(g(i), 1.0f, 1e-6f);
            ASSERT_NEAR(b(i), 0.0f, 1e-6f);
        }
    } END_TEST("Gamma initialized to ones, beta to zeros");

    TEST("Forward output shape") {
        CppNet::Layers::BatchNorm bn(3);
        Eigen::Tensor<float, 2> input(4, 3);
        input.setRandom();
        auto out = bn.forward(input);
        ASSERT_TRUE(out.dimension(0) == 4);
        ASSERT_TRUE(out.dimension(1) == 3);
    } END_TEST("Forward output shape");

    TEST("Training: output is approximately normalized") {
        CppNet::Layers::BatchNorm bn(3);
        bn.train();
        Eigen::Tensor<float, 2> input(32, 3);
        input.setRandom();
        auto out = bn.forward(input);
        // Check each feature has mean ≈ 0 and var ≈ 1
        for (int f = 0; f < 3; ++f) {
            float sum = 0, sum2 = 0;
            for (int b = 0; b < 32; ++b) { sum += out(b, f); sum2 += out(b, f)*out(b, f); }
            float mean = sum / 32.0f;
            float var = sum2 / 32.0f - mean * mean;
            ASSERT_NEAR(mean, 0.0f, 0.15f);
            ASSERT_NEAR(var, 1.0f, 0.3f);
        }
    } END_TEST("Training: output is approximately normalized");

    TEST("Running stats are updated during training") {
        CppNet::Layers::BatchNorm bn(2, 0.1f);
        bn.train();
        auto rm_before = bn.get_running_mean();
        Eigen::Tensor<float, 2> input(8, 2);
        input.setRandom();
        bn.forward(input);
        auto rm_after = bn.get_running_mean();
        bool changed = false;
        for (int i = 0; i < 2; ++i)
            if (std::fabs(rm_after(i) - rm_before(i)) > 1e-8f) changed = true;
        ASSERT_TRUE(changed);
    } END_TEST("Running stats are updated during training");

    TEST("Eval mode uses running stats") {
        CppNet::Layers::BatchNorm bn(2);
        bn.train();
        Eigen::Tensor<float, 2> input(16, 2);
        input.setRandom();
        // Run a few forward passes to build running stats
        for (int i = 0; i < 5; ++i) bn.forward(input);
        bn.eval();
        auto out = bn.forward(input);
        ASSERT_TRUE(out.dimension(0) == 16);
        ASSERT_TRUE(out.dimension(1) == 2);
        // Output should be finite
        for (int i = 0; i < out.size(); ++i)
            ASSERT_TRUE(std::isfinite(out.data()[i]));
    } END_TEST("Eval mode uses running stats");

    TEST("Backward shape matches") {
        CppNet::Layers::BatchNorm bn(3);
        bn.train();
        Eigen::Tensor<float, 2> input(4, 3);
        input.setRandom();
        bn.forward(input);
        Eigen::Tensor<float, 2> grad(4, 3);
        grad.setRandom();
        auto grad_in = bn.backward(grad);
        ASSERT_TRUE(grad_in.dimension(0) == 4);
        ASSERT_TRUE(grad_in.dimension(1) == 3);
    } END_TEST("Backward shape matches");

    TEST("Backward gradient is finite") {
        CppNet::Layers::BatchNorm bn(3);
        bn.train();
        Eigen::Tensor<float, 2> input(8, 3);
        input.setRandom();
        bn.forward(input);
        Eigen::Tensor<float, 2> grad(8, 3);
        grad.setRandom();
        auto grad_in = bn.backward(grad);
        for (int i = 0; i < grad_in.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad_in.data()[i]));
    } END_TEST("Backward gradient is finite");

    TEST("Freeze / unfreeze") {
        CppNet::Layers::BatchNorm bn(4);
        ASSERT_TRUE(bn.is_trainable());
        bn.freeze();
        ASSERT_TRUE(!bn.is_trainable());
        bn.unfreeze();
        ASSERT_TRUE(bn.is_trainable());
    } END_TEST("Freeze / unfreeze");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
