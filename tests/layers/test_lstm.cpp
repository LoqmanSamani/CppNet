/**
 * @file test_lstm.cpp
 * @brief Unit tests for the LSTM layer
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/layers/lstm.hpp"

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
    std::cout << "=== LSTM Tests ===\n";

    TEST("Construction") {
        CppNet::Layers::LSTM lstm(10, 20, true);
        ASSERT_TRUE(lstm.get_input_size() == 10);
        ASSERT_TRUE(lstm.get_hidden_size() == 20);
        ASSERT_TRUE(lstm.is_trainable());
    } END_TEST("Construction");

    TEST("Weight shapes") {
        CppNet::Layers::LSTM lstm(8, 16);
        ASSERT_TRUE(lstm.get_W_ih().dimension(0) == 8);
        ASSERT_TRUE(lstm.get_W_ih().dimension(1) == 64); // 4*16
        ASSERT_TRUE(lstm.get_W_hh().dimension(0) == 16);
        ASSERT_TRUE(lstm.get_W_hh().dimension(1) == 64);
        ASSERT_TRUE(lstm.get_bias().dimension(0) == 64);
    } END_TEST("Weight shapes");

    TEST("Forward output shape (return_sequences=true)") {
        CppNet::Layers::LSTM lstm(4, 8, true);
        Eigen::Tensor<float, 3> input(2, 5, 4);
        input.setRandom();
        auto out = lstm.forward(input);
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(1) == 5);
        ASSERT_TRUE(out.dimension(2) == 8);
    } END_TEST("Forward output shape (return_sequences=true)");

    TEST("Forward output shape (return_sequences=false)") {
        CppNet::Layers::LSTM lstm(4, 8, false);
        Eigen::Tensor<float, 3> input(2, 5, 4);
        input.setRandom();
        auto out = lstm.forward(input);
        ASSERT_TRUE(out.dimension(0) == 2);
        ASSERT_TRUE(out.dimension(2) == 8);
    } END_TEST("Forward output shape (return_sequences=false)");

    TEST("Output bounded by tanh (-1, 1)") {
        CppNet::Layers::LSTM lstm(4, 8, true);
        Eigen::Tensor<float, 3> input(2, 3, 4);
        input.setRandom();
        auto out = lstm.forward(input);
        for (int i = 0; i < out.size(); ++i) {
            ASSERT_TRUE(out.data()[i] >= -1.0f);
            ASSERT_TRUE(out.data()[i] <= 1.0f);
        }
    } END_TEST("Output bounded by tanh (-1, 1)");

    TEST("Output is finite") {
        CppNet::Layers::LSTM lstm(4, 8, true);
        Eigen::Tensor<float, 3> input(2, 3, 4);
        input.setRandom();
        auto out = lstm.forward(input);
        for (int i = 0; i < out.size(); ++i)
            ASSERT_TRUE(std::isfinite(out.data()[i]));
    } END_TEST("Output is finite");

    TEST("Backward shape") {
        CppNet::Layers::LSTM lstm(4, 8, true);
        Eigen::Tensor<float, 3> input(2, 3, 4);
        input.setRandom();
        lstm.forward(input);
        Eigen::Tensor<float, 3> grad(2, 3, 8);
        grad.setRandom();
        auto grad_in = lstm.backward(grad);
        ASSERT_TRUE(grad_in.dimension(0) == 2);
        ASSERT_TRUE(grad_in.dimension(1) == 3);
        ASSERT_TRUE(grad_in.dimension(2) == 4);
    } END_TEST("Backward shape");

    TEST("Backward is finite") {
        CppNet::Layers::LSTM lstm(4, 8, true);
        Eigen::Tensor<float, 3> input(2, 3, 4);
        input.setRandom();
        lstm.forward(input);
        Eigen::Tensor<float, 3> grad(2, 3, 8);
        grad.setRandom();
        auto grad_in = lstm.backward(grad);
        for (int i = 0; i < grad_in.size(); ++i)
            ASSERT_TRUE(std::isfinite(grad_in.data()[i]));
    } END_TEST("Backward is finite");

    TEST("Freeze / unfreeze") {
        CppNet::Layers::LSTM lstm(4, 8);
        lstm.freeze();
        ASSERT_TRUE(!lstm.is_trainable());
        lstm.unfreeze();
        ASSERT_TRUE(lstm.is_trainable());
    } END_TEST("Freeze / unfreeze");

    TEST("Single timestep") {
        CppNet::Layers::LSTM lstm(4, 8, true);
        Eigen::Tensor<float, 3> input(1, 1, 4);
        input.setRandom();
        auto out = lstm.forward(input);
        ASSERT_TRUE(out.dimension(1) == 1);
        ASSERT_TRUE(out.dimension(2) == 8);
    } END_TEST("Single timestep");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
