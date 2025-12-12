#ifndef INIT_HPP
#define INIT_HPP

#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace CppNet
{
    namespace Utils
    {
        class Initialization
        {
        public:

            Initialization(
                int in_size,
                int out_size,
                int channels_zero = 0,
                int channels_one = 0
            );

            // main initialization methods
            void init_params(Eigen::Tensor<float, 1>& params, const std::string& method);
            void init_params(Eigen::Tensor<float, 2>& params, const std::string& method);
            void init_params(Eigen::Tensor<float, 4>& params, const std::string& method);

            // internal structure to hold initialization configuration
            struct InitConfig
            {
                float scale;
                float mean;
                float std_dev;
                bool use_normal;
            };

            // static methods to compute initialization parameters
            static InitConfig get_xavier_uniform_config(int fan_in, int fan_out);
            static InitConfig get_xavier_normal_config(int fan_in, int fan_out);
            static InitConfig get_he_uniform_config(int fan_in);
            static InitConfig get_he_normal_config(int fan_in);
            static InitConfig get_lecun_uniform_config(int fan_in);
            static InitConfig get_lecun_normal_config(int fan_in);
            static InitConfig get_uniform_config();
            static InitConfig get_normal_config();

            // helper method to get config based on method name
            InitConfig get_config_for_method(const std::string& method) const;

            // helper method to apply initialization
            template<int Rank>
            void apply_initialization(Eigen::Tensor<float, Rank>& params, const InitConfig& config);

            private:
                int in_size_;
                int out_size_;
                int channels_zero_;
                int channels_one_;
        };
    }
}

#endif // INIT_HPP