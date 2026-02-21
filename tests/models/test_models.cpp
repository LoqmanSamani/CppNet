/**
 * @file test_models.cpp
 * @brief Unit tests for the SequentialModel
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CppNet/models/models.hpp"
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
    std::cout << "=== SequentialModel Tests ===\n";

    TEST("Empty model") {
        CppNet::Models::SequentialModel model;
        ASSERT_TRUE(model.num_layers() == 0);
    } END_TEST("Empty model");

    TEST("Add layers increases count") {
        CppNet::Models::SequentialModel model;
        model.add_layer(std::make_shared<CppNet::Layers::Linear>(4, 8, "l1", true, true, "cpu-eigen"));
        ASSERT_TRUE(model.num_layers() == 1);
        model.add_layer(std::make_shared<CppNet::Layers::Linear>(8, 2, "l2", true, true, "cpu-eigen"));
        ASSERT_TRUE(model.num_layers() == 2);
    } END_TEST("Add layers increases count");

    TEST("Get layer by index") {
        CppNet::Models::SequentialModel model;
        auto l1 = std::make_shared<CppNet::Layers::Linear>(4, 8, "l1", true, true, "cpu-eigen");
        auto l2 = std::make_shared<CppNet::Layers::Linear>(8, 2, "l2", true, true, "cpu-eigen");
        model.add_layer(l1);
        model.add_layer(l2);
        ASSERT_TRUE(model.get_layer(0) == l1);
        ASSERT_TRUE(model.get_layer(1) == l2);
    } END_TEST("Get layer by index");

    TEST("Summary does not throw") {
        CppNet::Models::SequentialModel model;
        model.add_layer(std::make_shared<CppNet::Layers::Linear>(10, 5, "l1", true, true, "cpu-eigen"));
        model.add_layer(std::make_shared<CppNet::Layers::Linear>(5, 1, "l2", true, true, "cpu-eigen"));
        model.summary(); // should print without crashing
        ASSERT_TRUE(true);
    } END_TEST("Summary does not throw");

    TEST("Get layer out of range") {
        CppNet::Models::SequentialModel model;
        model.add_layer(std::make_shared<CppNet::Layers::Linear>(4, 8, "l1", true, true, "cpu-eigen"));
        bool threw = false;
        try {
            model.get_layer(5);
        } catch (...) {
            threw = true;
        }
        ASSERT_TRUE(threw);
    } END_TEST("Get layer out of range");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
