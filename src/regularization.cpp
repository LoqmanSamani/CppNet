#include "regularization.hpp"

namespace CppNet
{
    namespace Regularizations
    {
        /************************************** L1 *************************************/
        L1::L1(double lambda) : lambda(lambda) {}

        void L1::apply(Eigen::Tensor<double, 2>& params)
        {
            // TODO: Implement L1 shrinkage
        }

        double L1::penalty(const Eigen::Tensor<double, 2>& params)
        {
            // TODO: Implement sum(|params|) * lambda
            return 0.0;
        }

        /************************************** L2 *************************************/
        L2::L2(double lambda) : lambda(lambda) {}

        void L2::apply(Eigen::Tensor<double, 2>& params)
        {
            // TODO: Implement L2 weight decay
        }

        double L2::penalty(const Eigen::Tensor<double, 2>& params)
        {
            // TODO: Implement sum(params²) * lambda
            return 0.0;
        }

        /************************************** ElasticNet *************************************/
        ElasticNet::ElasticNet(double l1, double l2) : l1(l1), l2(l2) {}

        void ElasticNet::apply(Eigen::Tensor<double, 2>& params)
        {
            // TODO: Implement ElasticNet update
        }

        double ElasticNet::penalty(const Eigen::Tensor<double, 2>& params)
        {
            // TODO: Implement l1*|params| + l2*params²
            return 0.0;
        }

        /************************************** Dropout *************************************/
        Dropout::Dropout(double rate) : rate(rate) {}

        void Dropout::apply(Eigen::Tensor<double, 2>& params)
        {
            // TODO: Implement dropout mask application
        }

        double Dropout::penalty(const Eigen::Tensor<double, 2>& params)
        {
            // Dropout usually doesn’t add explicit loss penalty
            return 0.0;
        }
    }
}
