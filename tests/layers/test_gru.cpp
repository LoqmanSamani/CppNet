/**
 * @file test_gru.cpp
 * @brief Unit tests for the GRU layer
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/gru.hpp"

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
    std::cout << "=== GRU Tests ===\n";

    TEST("Construction") {
        CppNet::Layers::GRU gru(10, 20, true);
        ASSERT_TRUE(gru.get_input_size() == 10);
        ASSERT_TRUE(gru.get_hidden_size() == 20);
        ASSERT_TRUE(gru.is_trainable());
    } END_TEST("Construction");

    TEST("Weight shapes") {
        CppNet::Layers::GRU gru(8, 16);
        ASSERT_TRUE(gru.get_W_ih().dimension(0) == 8);
        ASSERT_TRUE(gru.get_W_ih().dimension(1) == 48); // 3*16
        ASSERT_TRUE(gru.get_W_hh().dimension(0) == 16);
        ASSERT_TRUE(gru.get_W_hh().dimension(1) == 48);
        ASSERT_TRUE(gru.get_bias().dimension(0) == 48);
    } END_TEST("Weight shapes");

    TEST("Forward output shape (return_sequences=true)") {
        CppNet::Layers::GRU gru(4, 8, true);
        Eigen::Tensor<float, 3> input(2, 5, 4); // [batch, seq, input]
        input.setRandom();
        auto out = gru.forward(input);
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(1) == 5);
        ASSERT_TRUE(out.dimension(2) == 8);
    } END_TEST("Forward output shape (return_sequences=true)");

    TEST("Forward output shape (return_sequences=false)") {
        CppNet::Layers::GRU gru(4, 8, false);
        Eigen::Tensor<float, 3> input(2, 5, 4);
        input.setRandom();
        auto out = gru.forward(input);
        // return_sequences=false: only last timestep [batch, 1, hidden]
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(2) == 8);
    } END_TEST("Forward output shape (return_sequences=false)");

    TEST("Output is finite") {
        CppNet::Layers::GRU gru(4, 8, true);
        Eigen::Tensor<float, 3> input(2, 3, 4);
        input.setRandom();
        auto out = gru.forward(input);
        for (int i = 0; i < out.size(); ++i)
            ASSERT_TRUE(std::isfinite(out.data()[i]));
    } END_TEST("Output is finite");

    TEST("Backward shape") {
        CppNet::Layers::GRU gru(4, 8, true);
        Eigen::Tensor<float, 3> input(2, 3, 4);
        input.setRandom();
        auto out = gru.forward(input);
        Eigen::Tensor<float, 3> grad(2, 3, 8);
        grad.setRandom();
        auto grad_in = gru.backward(grad);
        ASSERT_TRUE(grad_in.dimension(0) == 2);
        ASSERT_TRUE(grad_in.dimension(1) == 3);
        ASSERT_TRUE(grad_in.dimension(2) == 4);
    } END_TEST("Backward shape");

    TEST("Backward is finite") {
        CppNet::Layers::GRU gru(4, 8, true);
        Eigen::Tensor<float, 3> input(2, 3, 4);
        input.setRandom();
        gru.forward(input);
        Eigen::Tensor<float, 3> grad(2, 3, 8);
        grad.setRandom();
        auto grad_in = gru.backward(grad);
        for (int i = 0; i < grad_in.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad_in.data()[i]));
    } END_TEST("Backward is finite");

    TEST("Freeze / unfreeze") {
        CppNet::Layers::GRU gru(4, 8);
        ASSERT_TRUE(gru.is_trainable());
        gru.freeze();
        ASSERT_TRUE(!gru.is_trainable());
        gru.unfreeze();
        ASSERT_TRUE(gru.is_trainable());
    } END_TEST("Freeze / unfreeze");

    TEST("Single timestep") {
        CppNet::Layers::GRU gru(4, 8, true);
        Eigen::Tensor<float, 3> input(1, 1, 4);
        input.setRandom();
        auto out = gru.forward(input);
        ASSERT_TRUE(out.dimension(1) == 1);
        ASSERT_TRUE(out.dimension(2) == 8);
    } END_TEST("Single timestep");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
