#include "optimizers.hpp"

namespace CppNet
{
    namespace Optimizers
    {
        /************************************** Optimizer Base *************************************/
        void Optimizer::zero_grad(std::vector<Eigen::Tensor<double, 2>>& grads)
        {
            for (auto& g : grads)
            {
                g.setZero();
            }
        }

        /************************************** SGD *************************************/
        SGD::SGD(double lr) : lr(lr) {}

        void SGD::step(std::vector<Eigen::Tensor<double, 2>>& params,
                       std::vector<Eigen::Tensor<double, 2>>& grads)
        {
            // TODO: Implement: params[i] -= lr * grads[i]
        }

        /************************************** Momentum *************************************/
        Momentum::Momentum(double lr, double momentum) : lr(lr), momentum(momentum) {}

        void Momentum::step(std::vector<Eigen::Tensor<double, 2>>& params,
                            std::vector<Eigen::Tensor<double, 2>>& grads)
        {
            // TODO: Implement Momentum update
        }

        /************************************** Adam *************************************/
        Adam::Adam(double lr, double beta1, double beta2, double eps)
            : lr(lr), beta1(beta1), beta2(beta2), eps(eps) {}

        void Adam::step(std::vector<Eigen::Tensor<double, 2>>& params,
                        std::vector<Eigen::Tensor<double, 2>>& grads)
        {
            // TODO: Implement Adam update
        }

        /************************************** RMSProp *************************************/
        RMSProp::RMSProp(double lr, double rho, double eps) : lr(lr), rho(rho), eps(eps) {}

        void RMSProp::step(std::vector<Eigen::Tensor<double, 2>>& params,
                           std::vector<Eigen::Tensor<double, 2>>& grads)
        {
            // TODO: Implement RMSProp update
        }

        /************************************** Adagrad *************************************/
        Adagrad::Adagrad(double lr, double eps) : lr(lr), eps(eps) {}

        void Adagrad::step(std::vector<Eigen::Tensor<double, 2>>& params,
                           std::vector<Eigen::Tensor<double, 2>>& grads)
        {
            // TODO: Implement Adagrad update
        }
    }
}