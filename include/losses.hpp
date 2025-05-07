#ifndef LOSSES_HPP
#define LOSSES_HPP

#include <Eigen/Dense>


namespace CppNet
{
    class BinaryCrossEntropy
    {
    public:
        // compute binary cross-entropy loss between true labels y and predictions y_hat
        double forward(const Eigen::MatrixXd& y, const Eigen::MatrixXd& y_hat);

        // compute gradient of loss with respect to y_hat
        Eigen::MatrixXd backward(const Eigen::MatrixXd& y_hat, const Eigen::MatrixXd& y);
    };
}



#endif // LOSSES_HPP






