#include "CppNet/regularizations/regularizations.hpp"




namespace CppNet
{
    namespace Regularizations
    {
        /************************************** L1 *************************************/
        L1::L1(float lambda) : lambda(lambda) {}

        void L1::apply(Eigen::Tensor<float, 2>& params)
        {
            // TODO: Implement L1 shrinkage
        }

        float L1::penalty(const Eigen::Tensor<float, 2>& params)
        {
            // TODO: Implement sum(|params|) * lambda
            return 0.0f;
        }

        /************************************** L2 *************************************/
        L2::L2(float lambda) : lambda(lambda) {}

        void L2::apply(Eigen::Tensor<float, 2>& params)
        {
            // TODO: Implement L2 weight decay
        }

        float L2::penalty(const Eigen::Tensor<float, 2>& params)
        {
            // TODO: Implement sum(params²) * lambda
            return 0.0f;
        }

        /************************************** ElasticNet *************************************/
        ElasticNet::ElasticNet(float l1, float l2) : l1(l1), l2(l2) {}

        void ElasticNet::apply(Eigen::Tensor<float, 2>& params)
        {
            // TODO: Implement ElasticNet update
        }

        float ElasticNet::penalty(const Eigen::Tensor<float, 2>& params)
        {
            // TODO: Implement l1*|params| + l2*params²
            return 0.0f;
        }

        /************************************** Dropout *************************************/
        Dropout::Dropout(float rate) : rate(rate) {}

        void Dropout::apply(Eigen::Tensor<float, 2>& params)
        {
            // TODO: Implement dropout mask application
        }

        float Dropout::penalty(const Eigen::Tensor<float, 2>& params)
        {
            // Dropout usually doesn’t add explicit loss penalty
            return 0.0f;
        }
    }
}
