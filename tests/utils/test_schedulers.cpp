/**
 * @file test_schedulers.cpp
 * @brief Unit tests for learning rate schedulers
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "CppNet/utils/schedulers.hpp"

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
    std::cout << "=== Scheduler Tests ===\n";

    // --- StepLR ---
    TEST("StepLR: initial LR") {
        CppNet::Schedulers::StepLR sched(0.1f, 5, 0.5f);
        ASSERT_NEAR(sched.get_lr(), 0.1f, 1e-6f);
        ASSERT_TRUE(sched.get_step_size() == 5);
        ASSERT_NEAR(sched.get_gamma(), 0.5f, 1e-6f);
    } END_TEST("StepLR: initial LR");

    TEST("StepLR: decays at step_size") {
        CppNet::Schedulers::StepLR sched(0.1f, 3, 0.5f);
        sched.step(); // epoch 1
        sched.step(); // epoch 2
        // After 2 steps: no decay yet (decays at epoch 3)
        sched.step(); // epoch 3 → decay
        float lr = sched.get_lr();
        ASSERT_NEAR(lr, 0.05f, 1e-5f); // 0.1 * 0.5
    } END_TEST("StepLR: decays at step_size");

    TEST("StepLR: multiple decays") {
        CppNet::Schedulers::StepLR sched(1.0f, 2, 0.1f);
        sched.step(); // epoch 1
        sched.step(); // epoch 2 → 0.1
        sched.step(); // epoch 3
        sched.step(); // epoch 4 → 0.01
        ASSERT_NEAR(sched.get_lr(), 0.01f, 1e-5f);
    } END_TEST("StepLR: multiple decays");

    TEST("StepLR: LR is always positive") {
        CppNet::Schedulers::StepLR sched(0.1f, 1, 0.5f);
        for (int i = 0; i < 20; ++i) sched.step();
        ASSERT_TRUE(sched.get_lr() > 0.0f);
    } END_TEST("StepLR: LR is always positive");

    // --- ExponentialLR ---
    TEST("ExponentialLR: initial LR") {
        CppNet::Schedulers::ExponentialLR sched(0.1f, 0.95f);
        ASSERT_NEAR(sched.get_lr(), 0.1f, 1e-6f);
        ASSERT_NEAR(sched.get_gamma(), 0.95f, 1e-6f);
    } END_TEST("ExponentialLR: initial LR");

    TEST("ExponentialLR: decays every step") {
        CppNet::Schedulers::ExponentialLR sched(1.0f, 0.5f);
        sched.step(); // epoch 1: 1.0 * 0.5 = 0.5
        ASSERT_NEAR(sched.get_lr(), 0.5f, 1e-5f);
        sched.step(); // epoch 2: 1.0 * 0.5^2 = 0.25
        ASSERT_NEAR(sched.get_lr(), 0.25f, 1e-5f);
    } END_TEST("ExponentialLR: decays every step");

    TEST("ExponentialLR: matches formula lr*gamma^epoch") {
        CppNet::Schedulers::ExponentialLR sched(0.1f, 0.9f);
        for (int i = 0; i < 10; ++i) sched.step();
        float expected = 0.1f * std::pow(0.9f, 10);
        ASSERT_NEAR(sched.get_lr(), expected, 1e-5f);
    } END_TEST("ExponentialLR: matches formula lr*gamma^epoch");

    // --- CosineAnnealingLR ---
    TEST("CosineAnnealingLR: initial LR") {
        CppNet::Schedulers::CosineAnnealingLR sched(0.1f, 50, 0.0f);
        ASSERT_NEAR(sched.get_lr(), 0.1f, 1e-6f);
        ASSERT_TRUE(sched.get_T_max() == 50);
        ASSERT_NEAR(sched.get_eta_min(), 0.0f, 1e-6f);
    } END_TEST("CosineAnnealingLR: initial LR");

    TEST("CosineAnnealingLR: reaches eta_min at T_max") {
        CppNet::Schedulers::CosineAnnealingLR sched(1.0f, 10, 0.0f);
        for (int i = 0; i < 10; ++i) sched.step();
        ASSERT_NEAR(sched.get_lr(), 0.0f, 1e-4f);
    } END_TEST("CosineAnnealingLR: reaches eta_min at T_max");

    TEST("CosineAnnealingLR: at T_max/2 is roughly midpoint") {
        CppNet::Schedulers::CosineAnnealingLR sched(1.0f, 10, 0.0f);
        for (int i = 0; i < 5; ++i) sched.step();
        float lr = sched.get_lr();
        // At epoch 5/10: eta_min + 0.5*(lr0-eta_min)*(1+cos(pi*5/10)) = 0.5*(1+0)=0.5
        ASSERT_NEAR(lr, 0.5f, 0.1f);
    } END_TEST("CosineAnnealingLR: at T_max/2 is roughly midpoint");

    TEST("CosineAnnealingLR: LR decreases monotonically to T_max") {
        CppNet::Schedulers::CosineAnnealingLR sched(1.0f, 20, 0.0f);
        float prev = sched.get_lr();
        for (int i = 0; i < 20; ++i) {
            sched.step();
            float cur = sched.get_lr();
            ASSERT_TRUE(cur <= prev + 1e-5f);
            prev = cur;
        }
    } END_TEST("CosineAnnealingLR: LR decreases monotonically to T_max");

    TEST("CosineAnnealingLR: non-zero eta_min") {
        CppNet::Schedulers::CosineAnnealingLR sched(1.0f, 10, 0.01f);
        for (int i = 0; i < 10; ++i) sched.step();
        ASSERT_NEAR(sched.get_lr(), 0.01f, 1e-4f);
    } END_TEST("CosineAnnealingLR: non-zero eta_min");

    std::cout << "\n=== Results: " << tests_passed << " passed, " << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
