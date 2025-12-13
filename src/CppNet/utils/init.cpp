#include "CppNet/utils/init.hpp"



namespace CppNet
{
    namespace Utils
    {
        Initialization::Initialization(
            int in_size,
            int out_size,
            int channels_zero,
            int channels_one
        ):
        in_size_(in_size), 
        out_size_(out_size), 
        channels_zero_(channels_zero), 
        channels_one_(channels_one)
        {}

        Initialization::InitConfig Initialization::get_xavier_uniform_config(int fan_in, int fan_out)
        {
            // Xavier/Glorot uniform: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
            InitConfig config;
            config.scale = std::sqrt(6.0f / (fan_in + fan_out));
            config.mean = 0.0f;
            config.std_dev = 0.0f;
            config.use_normal = false;
            return config;
        }

        Initialization::InitConfig Initialization::get_xavier_normal_config(int fan_in, int fan_out)
        {
            // Xavier/Glorot normal: N(0, sqrt(2/(fan_in + fan_out)))
            InitConfig config;
            config.scale = 0.0f;
            config.mean = 0.0f;
            config.std_dev = std::sqrt(2.0f / (fan_in + fan_out));
            config.use_normal = true;
            return config;
        }

        Initialization::InitConfig Initialization::get_he_uniform_config(int fan_in)
        {
            // He uniform (for ReLU): U(-sqrt(6/fan_in), sqrt(6/fan_in))
            InitConfig config;
            config.scale = std::sqrt(6.0f / fan_in);
            config.mean = 0.0f;
            config.std_dev = 0.0f;
            config.use_normal = false;
            return config;
        }

        Initialization::InitConfig Initialization::get_he_normal_config(int fan_in)
        {
            // He normal (for ReLU): N(0, sqrt(2/fan_in))
            InitConfig config;
            config.scale = 0.0f;
            config.mean = 0.0f;
            config.std_dev = std::sqrt(2.0f / fan_in);
            config.use_normal = true;
            return config;
        }

        Initialization::InitConfig Initialization::get_lecun_uniform_config(int fan_in)
        {
            // LeCun uniform: U(-sqrt(3/fan_in), sqrt(3/fan_in))
            InitConfig config;
            config.scale = std::sqrt(3.0f / fan_in);
            config.mean = 0.0f;
            config.std_dev = 0.0f;
            config.use_normal = false;
            return config;
        }

        Initialization::InitConfig Initialization::get_lecun_normal_config(int fan_in)
        {
            // LeCun normal: N(0, sqrt(1/fan_in))
            InitConfig config;
            config.scale = 0.0f;
            config.mean = 0.0f;
            config.std_dev = std::sqrt(1.0f / fan_in);
            config.use_normal = true;
            return config;
        }

        Initialization::InitConfig Initialization::get_uniform_config()
        {
            // simple uniform distribution: U(-0.1, 0.1)
            InitConfig config;
            config.scale = 0.1f;
            config.mean = 0.0f;
            config.std_dev = 0.0f;
            config.use_normal = false;
            return config;
        }

        Initialization::InitConfig Initialization::get_normal_config()
        {
            // simple normal distribution: N(0, 0.01)
            InitConfig config;
            config.scale = 0.0f;
            config.mean = 0.0f;
            config.std_dev = 0.01f;
            config.use_normal = true;
            return config;
        }

        Initialization::InitConfig Initialization::get_config_for_method(const std::string& method) const
        {
            if (method == "xavier")
            {
                return get_xavier_uniform_config(in_size_, out_size_);
            }
            else if (method == "xavier_normal")
            {
                return get_xavier_normal_config(in_size_, out_size_);
            }
            else if (method == "he_normal")
            {
                return get_he_normal_config(in_size_);
            }
            else if (method == "he")
            {
                return get_he_uniform_config(in_size_);
            }
            else if (method == "lecun_normal")
            {
                return get_lecun_normal_config(in_size_);
            }
            else if (method == "lecun")
            {
                return get_lecun_uniform_config(in_size_);
            }
            else if (method == "uniform")
            {
                return get_uniform_config();
            }
            else if (method == "normal")
            {
                return get_normal_config();
            }
            else
            {
                throw std::runtime_error("Unknown weight initialization method: '" + method + 
                                        "'\nSupported methods: xavier, xavier_normal, he, he_normal, " +
                                        "lecun, lecun_normal, uniform, normal, zeros, ones");
            }
        }

        template<int Rank>
        void Initialization::apply_initialization(Eigen::Tensor<float, Rank>& params, const InitConfig& config)
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto base_seed = static_cast<unsigned>(now.time_since_epoch().count());

            #pragma omp parallel
            {
                // create thread-local random number generator with unique seed
                unsigned thread_seed = base_seed + static_cast<unsigned>(omp_get_thread_num()) * 1000000;
                std::mt19937 local_gen(thread_seed);

                if (config.use_normal)
                {
                    std::normal_distribution<float> local_dist(config.mean, config.std_dev);

                    #pragma omp for
                    for (int idx = 0; idx < params.size(); ++idx)
                    {
                        params.data()[idx] = local_dist(local_gen);
                    }
                }
                else
                {
                    std::uniform_real_distribution<float> local_dist(-config.scale, config.scale);

                    #pragma omp for
                    for (int idx = 0; idx < params.size(); ++idx)
                    {
                        params.data()[idx] = local_dist(local_gen);
                    }
                }
            }
        }

        void Initialization::init_params(Eigen::Tensor<float, 1>& params, const std::string& method)
        {
            params = Eigen::Tensor<float, 1>(out_size_);

            if (method == "zeros")
            {
                params.setZero();
                return;
            }
            else if (method == "ones")
            {
                params.setConstant(1.0f);
                return;
            }

            InitConfig config = get_config_for_method(method);
            apply_initialization(params, config);
        }

        void Initialization::init_params(Eigen::Tensor<float, 2>& params, const std::string& method)
        {
            params = Eigen::Tensor<float, 2>(in_size_, out_size_);

            if (method == "zeros")
            {
                params.setZero();
                return;
            }
            else if (method == "ones")
            {
                params.setConstant(1.0f);
                return;
            }

            InitConfig config = get_config_for_method(method);
            apply_initialization(params, config);
        }

        void Initialization::init_params(Eigen::Tensor<float, 4>& params, const std::string& method)
        {
            params = Eigen::Tensor<float, 4>(out_size_, in_size_, channels_zero_, channels_one_);

            if (method == "zeros")
            {
                params.setZero();
                return;
            }
            else if (method == "ones")
            {
                params.setConstant(1.0f);
                return;
            }

            InitConfig config = get_config_for_method(method);
            apply_initialization(params, config);
        }
    }
}