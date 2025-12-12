#include "CppNet/utils/init.hpp"
#include <iostream>
#include <cmath>
#include <cassert>





void test_initialization()
{
    std::cout << "starting Initialization tests...\n\n";
    
    int in_size = 100;
    int out_size = 50;
    int channels_zero = 3;
    int channels_one = 3;
    
    CppNet::Utils::Initialization init(in_size, out_size, channels_zero, channels_one);
    
    std::cout << "test 1: Zeros initialization\n";
    {
        Eigen::Tensor<float, 1> params1d;
        init.init_params(params1d, "zeros");
        
        bool all_zeros = true;
        for (int i = 0; i < params1d.size(); ++i)
        {
            if (params1d.data()[i] != 0.0f)
            {
                all_zeros = false;
                break;
            }
        }
        assert(all_zeros && "1d zeros initialization failed");
        std::cout << "  1d tensor: PASSED\n";
        
        Eigen::Tensor<float, 2> params2d;
        init.init_params(params2d, "zeros");
        
        all_zeros = true;
        for (int i = 0; i < params2d.size(); ++i)
        {
            if (params2d.data()[i] != 0.0f)
            {
                all_zeros = false;
                break;
            }
        }
        assert(all_zeros && "2d zeros initialization failed");
        std::cout << "  2d tensor: PASSED\n";
    }
    
    std::cout << "\ntest 2: ones initialization\n";
    {
        Eigen::Tensor<float, 1> params1d;
        init.init_params(params1d, "ones");
        
        bool all_ones = true;
        for (int i = 0; i < params1d.size(); ++i)
        {
            if (params1d.data()[i] != 1.0f)
            {
                all_ones = false;
                break;
            }
        }
        assert(all_ones && "1d ones initialization failed");
        std::cout << "  1d tensor: PASSED\n";
        
        Eigen::Tensor<float, 2> params2d;
        init.init_params(params2d, "ones");
        
        all_ones = true;
        for (int i = 0; i < params2d.size(); ++i)
        {
            if (params2d.data()[i] != 1.0f)
            {
                all_ones = false;
                break;
            }
        }
        assert(all_ones && "2d ones initialization failed");
        std::cout << "  2d tensor: PASSED\n";
    }
    
    std::cout << "\ntest 3: xavier uniform initialization\n";
    {
        Eigen::Tensor<float, 2> params;
        init.init_params(params, "xavier");
        
        float expected_scale = std::sqrt(6.0f / (in_size + out_size));
        
        bool in_range = true;
        for (int i = 0; i < params.size(); ++i)
        {
            if (std::abs(params.data()[i]) > expected_scale)
            {
                in_range = false;
                break;
            }
        }
        assert(in_range && "xavier uniform values out of range");
        
        bool has_variation = false;
        float first_val = params.data()[0];
        for (int i = 1; i < params.size(); ++i)
        {
            if (params.data()[i] != first_val)
            {
                has_variation = true;
                break;
            }
        }
        assert(has_variation && "xavier uniform has no variation");
        std::cout << "  values in range [-" << expected_scale << ", " << expected_scale << "]: PASSED\n";
    }
    
    std::cout << "\ntest 4: he normal initialization\n";
    {
        Eigen::Tensor<float, 2> params;
        init.init_params(params, "he_normal");
        
        float sum = 0.0f;
        for (int i = 0; i < params.size(); ++i)
        {
            sum += params.data()[i];
        }
        float mean = sum / params.size();
        
        float variance_sum = 0.0f;
        for (int i = 0; i < params.size(); ++i)
        {
            float diff = params.data()[i] - mean;
            variance_sum += diff * diff;
        }
        float std_dev = std::sqrt(variance_sum / params.size());
        
        float expected_std = std::sqrt(2.0f / in_size);
        
        assert(std::abs(mean) < 0.1f && "he normal mean not close to 0");
        
        float std_ratio = std_dev / expected_std;
        assert(std_ratio > 0.8f && std_ratio < 1.2f && "he normal std dev out of range");
        
        std::cout << "  mean: " << mean << " (expected ~0.0)\n";
        std::cout << "  std dev: " << std_dev << " (expected ~" << expected_std << ")\n";
        std::cout << "  PASSED\n";
    }
    
    std::cout << "\ntest 5: xavier normal initialization\n";
    {
        Eigen::Tensor<float, 2> params;
        init.init_params(params, "xavier_normal");
        
        float sum = 0.0f;
        for (int i = 0; i < params.size(); ++i)
        {
            sum += params.data()[i];
        }
        float mean = sum / params.size();
        
        float variance_sum = 0.0f;
        for (int i = 0; i < params.size(); ++i)
        {
            float diff = params.data()[i] - mean;
            variance_sum += diff * diff;
        }
        float std_dev = std::sqrt(variance_sum / params.size());
        
        float expected_std = std::sqrt(2.0f / (in_size + out_size));
        
        float std_ratio = std_dev / expected_std;
        assert(std_ratio > 0.8f && std_ratio < 1.2f && "xavier normal std dev out of range");
        
        std::cout << "  std dev: " << std_dev << " (expected ~" << expected_std << "): PASSED\n";
    }
    
    std::cout << "\ntest 6: He uniform initialization\n";
    {
        Eigen::Tensor<float, 2> params;
        init.init_params(params, "he_uniform");
        
        float expected_scale = std::sqrt(6.0f / in_size);
        
        bool in_range = true;
        for (int i = 0; i < params.size(); ++i)
        {
            if (std::abs(params.data()[i]) > expected_scale)
            {
                in_range = false;
                break;
            }
        }
        assert(in_range && "he uniform values out of range");
        std::cout << "  values in range: PASSED\n";
    }
    
    std::cout << "\ntest 7: lecun normal initialization\n";
    {
        Eigen::Tensor<float, 2> params;
        init.init_params(params, "lecun");
        
        float sum = 0.0f;
        for (int i = 0; i < params.size(); ++i)
        {
            sum += params.data()[i];
        }
        float mean = sum / params.size();
        
        float variance_sum = 0.0f;
        for (int i = 0; i < params.size(); ++i)
        {
            float diff = params.data()[i] - mean;
            variance_sum += diff * diff;
        }
        float std_dev = std::sqrt(variance_sum / params.size());
        
        float expected_std = std::sqrt(1.0f / in_size);
        
        float std_ratio = std_dev / expected_std;
        assert(std_ratio > 0.8f && std_ratio < 1.2f && "lecun normal std dev out of range");
        std::cout << "  std dev: " << std_dev << " (expected ~" << expected_std << "): PASSED\n";
    }
    
    std::cout << "\ntest 8: lecun uniform initialization\n";
    {
        Eigen::Tensor<float, 2> params;
        init.init_params(params, "lecun_uniform");
        
        float expected_scale = std::sqrt(3.0f / in_size);
        
        bool in_range = true;
        for (int i = 0; i < params.size(); ++i)
        {
            if (std::abs(params.data()[i]) > expected_scale)
            {
                in_range = false;
                break;
            }
        }
        assert(in_range && "lecun uniform values out of range");
        std::cout << "  values in range: PASSED\n";
    }
    
    std::cout << "\ntest 9: simple uniform initialization\n";
    {
        Eigen::Tensor<float, 2> params;
        init.init_params(params, "uniform");
        
        bool in_range = true;
        for (int i = 0; i < params.size(); ++i)
        {
            if (std::abs(params.data()[i]) > 0.1f)
            {
                in_range = false;
                break;
            }
        }
        assert(in_range && "uniform values out of range");
        std::cout << "  values in range [-0.1, 0.1]: PASSED\n";
    }

    std::cout << "\ntest 10: simple normal initialization\n";
    {
        Eigen::Tensor<float, 2> params;
        init.init_params(params, "normal");
        
        float sum = 0.0f;
        for (int i = 0; i < params.size(); ++i)
        {
            sum += params.data()[i];
        }
        float mean = sum / params.size();
        
        float variance_sum = 0.0f;
        for (int i = 0; i < params.size(); ++i)
        {
            float diff = params.data()[i] - mean;
            variance_sum += diff * diff;
        }
        float std_dev = std::sqrt(variance_sum / params.size());
        
        float std_ratio = std_dev / 0.01f;
        assert(std_ratio > 0.8f && std_ratio < 1.2f && "normal std dev out of range");
        std::cout << "  std dev: " << std_dev << " (expected ~0.01): PASSED\n";
    }
    
    std::cout << "\ntest 11: 4d tensor initialization\n";
    {
        Eigen::Tensor<float, 4> params;
        init.init_params(params, "he_normal");
        
        assert(params.dimension(0) == out_size && "4d tensor dim 0 incorrect");
        assert(params.dimension(1) == in_size && "4d tensor dim 1 incorrect");
        assert(params.dimension(2) == channels_zero && "4d tensor dim 2 incorrect");
        assert(params.dimension(3) == channels_one && "4d tensor dim 3 incorrect");
        
        bool has_variation = false;
        float first_val = params.data()[0];
        for (int i = 1; i < params.size(); ++i)
        {
            if (params.data()[i] != first_val)
            {
                has_variation = true;
                break;
            }
        }
        assert(has_variation && "4d tensor has no variation");
        std::cout << "  dimensions and values: PASSED\n";
    }
    
    std::cout << "\ntest 12: invalid method error handling\n";
    {
        Eigen::Tensor<float, 2> params;
        bool caught_exception = false;
        try
        {
            init.init_params(params, "invalid_method");
        }
        catch (const std::runtime_error& e)
        {
            caught_exception = true;
        }
        assert(caught_exception && "failed to catch invalid method exception");
        std::cout << "  exception handling: PASSED\n";
    }
    
    std::cout << "\ntest 13: randomness check\n";
    {
        Eigen::Tensor<float, 2> params1;
        Eigen::Tensor<float, 2> params2;
        
        init.init_params(params1, "xavier");
        init.init_params(params2, "xavier");
        
        bool are_different = false;
        for (int i = 0; i < params1.size(); ++i)
        {
            if (params1.data()[i] != params2.data()[i])
            {
                are_different = true;
                break;
            }
        }
        assert(are_different && "multiple initializations produce identical results");
        std::cout << "  different random values: PASSED\n";
    }
    
    std::cout << "\n========================================\n";
    std::cout << "ALL TESTS PASSED!\n";
    std::cout << "========================================\n";
}

int main()
{
    test_initialization();
    return 0;
}