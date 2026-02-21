/**
 * @file test_residual.cpp
 * @brief Unit tests for the Residual (skip-connection) layer
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/residual.hpp"
#include "CppNet/layers/linear.hpp"

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
    std::cout << "=== Residual Tests ===\n";

    TEST("Construction (same dims - no projection)") {
        auto layer = std::make_shared<CppNet::Layers::Linear>(8, 8, "l1", true, true, "cpu-eigen");
        std::vector<std::shared_ptr<CppNet::Layers::Layer>> block = {layer};
        CppNet::Layers::Residual res(block, 8, 8);
        ASSERT_TRUE(!res.has_projection());
        ASSERT_TRUE(res.num_block_layers() == 1);
    } END_TEST("Construction (same dims - no projection)");

    TEST("Construction (different dims - with projection)") {
        auto layer = std::make_shared<CppNet::Layers::Linear>(8, 16, "l1", true, true, "cpu-eigen");
        std::vector<std::shared_ptr<CppNet::Layers::Layer>> block = {layer};
        CppNet::Layers::Residual res(block, 8, 16);
        ASSERT_TRUE(res.has_projection());
    } END_TEST("Construction (different dims - with projection)");

    TEST("Forward shape (same dims)") {
        auto layer = std::make_shared<CppNet::Layers::Linear>(4, 4, "l1", true, true, "cpu-eigen");
        std::vector<std::shared_ptr<CppNet::Layers::Layer>> block = {layer};
        CppNet::Layers::Residual res(block, 4, 4);

        Eigen::Tensor<float, 2> input(2, 4);
        input.setRandom();
        auto out = res.forward(input);
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(1) == 4);
    } END_TEST("Forward shape (same dims)");

    TEST("Forward shape (different dims with projection)") {
        auto layer = std::make_shared<CppNet::Layers::Linear>(4, 8, "l1", true, true, "cpu-eigen");
        std::vector<std::shared_ptr<CppNet::Layers::Layer>> block = {layer};
        CppNet::Layers::Residual res(block, 4, 8);

        Eigen::Tensor<float, 2> input(2, 4);
        input.setRandom();
        auto out = res.forward(input);
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(1) == 8);
    } END_TEST("Forward shape (different dims with projection)");

    TEST("Skip connection adds input") {
        // Identity block: zero weights so block output = just bias
        auto layer = std::make_shared<CppNet::Layers::Linear>(2, 2, "l1", true, false, "cpu-eigen");
        // Set weights to zero, so block(x) ≈ 0
        Eigen::Tensor<float, 2> zero_w(2, 2);
        zero_w.setZero();
        layer->set_weights(zero_w);
        std::vector<std::shared_ptr<CppNet::Layers::Layer>> block = {layer};
        CppNet::Layers::Residual res(block, 2, 2);

        Eigen::Tensor<float, 2> input(1, 2);
        input.setValues({{3.0f, 7.0f}});
        auto out = res.forward(input);
        // output ≈ 0 + input = input
        ASSERT_NEAR(out(0, 0), 3.0f, 0.1f);
        ASSERT_NEAR(out(0, 1), 7.0f, 0.1f);
    } END_TEST("Skip connection adds input");

    TEST("Output is finite") {
        auto layer = std::make_shared<CppNet::Layers::Linear>(4, 4, "l1", true, true, "cpu-eigen");
        std::vector<std::shared_ptr<CppNet::Layers::Layer>> block = {layer};
        CppNet::Layers::Residual res(block, 4, 4);

        Eigen::Tensor<float, 2> input(3, 4);
        input.setRandom();
        auto out = res.forward(input);
        for (int i = 0; i < out.size(); ++i)
            ASSERT_TRUE(std::isfinite(out.data()[i]));
    } END_TEST("Output is finite");

    TEST("Backward shape") {
        auto layer = std::make_shared<CppNet::Layers::Linear>(4, 4, "l1", true, true, "cpu-eigen");
        std::vector<std::shared_ptr<CppNet::Layers::Layer>> block = {layer};
        CppNet::Layers::Residual res(block, 4, 4);

        Eigen::Tensor<float, 2> input(2, 4);
        input.setRandom();
        res.forward(input);
        Eigen::Tensor<float, 2> grad(2, 4);
        grad.setRandom();
        auto grad_in = res.backward(grad);
        ASSERT_TRUE(grad_in.dimension(0) == 2);
        ASSERT_TRUE(grad_in.dimension(1) == 4);
    } END_TEST("Backward shape");

    TEST("Is trainable") {
        auto layer = std::make_shared<CppNet::Layers::Linear>(4, 4, "l1", true, true, "cpu-eigen");
        std::vector<std::shared_ptr<CppNet::Layers::Layer>> block = {layer};
        CppNet::Layers::Residual res(block, 4, 4);
        ASSERT_TRUE(res.is_trainable());
    } END_TEST("Is trainable");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
