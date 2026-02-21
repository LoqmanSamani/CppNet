/**
 * @file test_embedding.cpp
 * @brief Unit tests for the Embedding layer
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/embedding.hpp"

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
    std::cout << "=== Embedding Tests ===\n";

    TEST("Construction") {
        CppNet::Layers::Embedding emb(100, 32);
        ASSERT_TRUE(emb.get_vocab_size() == 100);
        ASSERT_TRUE(emb.get_embed_dim() == 32);
        ASSERT_TRUE(emb.is_trainable());
    } END_TEST("Construction");

    TEST("Weight shape") {
        CppNet::Layers::Embedding emb(50, 16);
        auto& w = emb.get_weight();
        ASSERT_TRUE(w.dimension(0) == 50);
        ASSERT_TRUE(w.dimension(1) == 16);
    } END_TEST("Weight shape");

    TEST("Forward output shape") {
        CppNet::Layers::Embedding emb(100, 32);
        Eigen::Tensor<int, 2> input(2, 5);
        input.setValues({{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}});
        auto out = emb.forward(input);
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(1) == 5);
        ASSERT_TRUE(out.dimension(2) == 32);
    } END_TEST("Forward output shape");

    TEST("Lookup correctness") {
        CppNet::Layers::Embedding emb(10, 4);
        auto& w = emb.get_weight();
        // Set specific row
        for (int j = 0; j < 4; ++j) w(3, j) = (float)(j + 1);
        Eigen::Tensor<int, 2> input(1, 1);
        input(0, 0) = 3;
        auto out = emb.forward(input);
        for (int j = 0; j < 4; ++j)
            ASSERT_NEAR(out(0, 0, j), (float)(j + 1), 1e-6f);
    } END_TEST("Lookup correctness");

    TEST("Backward shape") {
        CppNet::Layers::Embedding emb(100, 16);
        Eigen::Tensor<int, 2> input(2, 3);
        input.setValues({{1, 2, 3}, {4, 5, 6}});
        emb.forward(input);
        Eigen::Tensor<float, 3> grad(2, 3, 16);
        grad.setRandom();
        auto grad_out = emb.backward(grad);
        ASSERT_TRUE(grad_out.dimension(0) == 2);
        ASSERT_TRUE(grad_out.dimension(1) == 3);
        ASSERT_TRUE(grad_out.dimension(2) == 16);
    } END_TEST("Backward shape");

    TEST("Freeze / unfreeze") {
        CppNet::Layers::Embedding emb(50, 8);
        ASSERT_TRUE(emb.is_trainable());
        emb.freeze();
        ASSERT_TRUE(!emb.is_trainable());
        emb.unfreeze();
        ASSERT_TRUE(emb.is_trainable());
    } END_TEST("Freeze / unfreeze");

    TEST("Output is finite") {
        CppNet::Layers::Embedding emb(20, 8);
        Eigen::Tensor<int, 2> input(3, 4);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                input(i, j) = (i * 4 + j) % 20;
        auto out = emb.forward(input);
        for (int i = 0; i < out.size(); ++i)
            ASSERT_TRUE(std::isfinite(out.data()[i]));
    } END_TEST("Output is finite");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
