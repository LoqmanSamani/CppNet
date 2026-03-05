/**
 * @file test_matmul_gpu.cu
 * @brief GPU matmul kernel tests via Linear layer forward/backward on CUDA
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cuda_runtime.h>
#include "CppNet/layers/linear.hpp"
#include "CppNet/optimizers/sgd.hpp"

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
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cout << "No CUDA devices found — skipping GPU matmul tests." << std::endl;
        return 0;
    }

    std::cout << "=== GPU Matmul Kernel Tests (via Linear layer) ===" << std::endl;

    TEST("GPU forward - output shape")
    {
        CppNet::Layers::Linear layer(10, 5, "fc_gpu", true, true, "gpu", "xavier");
        layer.set_max_batch_size(4);
        Eigen::Tensor<float, 2> input(4, 10);
        input.setRandom();
        auto output = layer.forward(input);
        ASSERT_TRUE(output.dimension(0) == 4);
        ASSERT_TRUE(output.dimension(1) == 5);
    }
    END_TEST("GPU forward - output shape");

    TEST("GPU forward - identity weight check")
    {
        CppNet::Layers::Linear layer(3, 3, "fc_id", true, false, "gpu", "xavier");
        Eigen::Tensor<float, 2> w(3, 3);
        w.setZero();
        w(0, 0) = 1.0f; w(1, 1) = 1.0f; w(2, 2) = 1.0f;
        layer.set_weights(w);
        layer.set_max_batch_size(2);

        Eigen::Tensor<float, 2> input(2, 3);
        input.setValues({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
        auto output = layer.forward(input);

        ASSERT_NEAR(output(0, 0), 1.0f, 1e-3f);
        ASSERT_NEAR(output(0, 1), 2.0f, 1e-3f);
        ASSERT_NEAR(output(0, 2), 3.0f, 1e-3f);
        ASSERT_NEAR(output(1, 0), 4.0f, 1e-3f);
        ASSERT_NEAR(output(1, 1), 5.0f, 1e-3f);
        ASSERT_NEAR(output(1, 2), 6.0f, 1e-3f);
    }
    END_TEST("GPU forward - identity weight check");

    TEST("GPU forward with bias")
    {
        CppNet::Layers::Linear layer(2, 2, "fc_bias", true, true, "gpu", "xavier");
        Eigen::Tensor<float, 2> w(2, 2);
        w.setValues({{1.0f, 0.0f}, {0.0f, 1.0f}});
        layer.set_weights(w);
        Eigen::Tensor<float, 1> b(2);
        b.setValues({0.5f, -0.5f});
        layer.set_biases(b);
        layer.set_max_batch_size(1);

        Eigen::Tensor<float, 2> input(1, 2);
        input.setValues({{3.0f, 4.0f}});
        auto output = layer.forward(input);

        ASSERT_NEAR(output(0, 0), 3.5f, 1e-3f);
        ASSERT_NEAR(output(0, 1), 3.5f, 1e-3f);
    }
    END_TEST("GPU forward with bias");

    TEST("GPU backward - gradient shape")
    {
        CppNet::Layers::Linear layer(6, 4, "fc_bwd", true, true, "gpu", "xavier");
        layer.set_max_batch_size(2);
        Eigen::Tensor<float, 2> input(2, 6);
        input.setRandom();
        layer.forward(input);

        Eigen::Tensor<float, 2> grad_output(2, 4);
        grad_output.setConstant(1.0f);
        auto grad_input = layer.backward(grad_output);

        ASSERT_TRUE(grad_input.dimension(0) == 2);
        ASSERT_TRUE(grad_input.dimension(1) == 6);
    }
    END_TEST("GPU backward - gradient shape");

    TEST("GPU step updates weights")
    {
        CppNet::Layers::Linear layer(4, 2, "fc_step", true, true, "gpu", "xavier");
        layer.set_max_batch_size(1);
        Eigen::Tensor<float, 2> input(1, 4);
        input.setConstant(1.0f);
        layer.forward(input);

        Eigen::Tensor<float, 2> grad(1, 2);
        grad.setConstant(1.0f);
        layer.backward(grad);

        Eigen::Tensor<float, 2> w_before = layer.get_weights();
        CppNet::Optimizers::SGD optimizer;
        layer.step(optimizer, 0.01f);
        auto& w_after = layer.get_weights();

        bool changed = false;
        for (int i = 0; i < 4 && !changed; ++i)
            for (int j = 0; j < 2 && !changed; ++j)
                if (std::fabs(w_before(i, j) - w_after(i, j)) > 1e-8f)
                    changed = true;
        ASSERT_TRUE(changed);
    }
    END_TEST("GPU step updates weights");

    TEST("GPU vs CPU results match")
    {
        CppNet::Layers::Linear gpu_layer(8, 4, "fc_gpu", true, true, "gpu", "xavier");
        CppNet::Layers::Linear cpu_layer(8, 4, "fc_cpu", true, true, "cpu-eigen", "xavier");

        // Copy GPU layer's weights to CPU layer
        cpu_layer.set_weights(gpu_layer.get_weights());
        cpu_layer.set_biases(gpu_layer.get_biases());
        gpu_layer.set_max_batch_size(3);

        Eigen::Tensor<float, 2> input(3, 8);
        input.setRandom();

        auto gpu_out = gpu_layer.forward(input);
        auto cpu_out = cpu_layer.forward(input);

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                ASSERT_NEAR(gpu_out(i, j), cpu_out(i, j), 1e-2f);
    }
    END_TEST("GPU vs CPU results match");

    TEST("GPU large batch matmul")
    {
        CppNet::Layers::Linear layer(64, 32, "fc_large", true, true, "gpu", "xavier");
        layer.set_max_batch_size(128);
        Eigen::Tensor<float, 2> input(128, 64);
        input.setRandom();
        auto output = layer.forward(input);
        ASSERT_TRUE(output.dimension(0) == 128);
        ASSERT_TRUE(output.dimension(1) == 32);

        // Check output is finite
        bool all_finite = true;
        for (int i = 0; i < 128 && all_finite; ++i)
            for (int j = 0; j < 32 && all_finite; ++j)
                if (!std::isfinite(output(i, j)))
                    all_finite = false;
        ASSERT_TRUE(all_finite);
    }
    END_TEST("GPU large batch matmul");

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
