#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


using namespace std;


int main()
{
    Eigen::Tensor<float, 3> a(3, 2, 5);
    a.setRandom();
    Eigen::Tensor<float, 3> b(3, 5, 3);
    b.setRandom();

    Eigen::array<Eigen::IndexPair<int>, 1> contraction_pair = {Eigen::IndexPair<int>(2, 1)};
    Eigen::Tensor<float, 4> c = a.contract(b, contraction_pair);
    
    std::cout << a.dimensions();
    std::cout << b.dimensions();
    std::cout << c.dimensions();

    const auto& d = c.dimensions();
    cout << "Dim Size is " << c.size() <<  " dim 0: " << d[0] << " dim 1: " << d[1]  << " dim 2: " << d[2]  << " dim 3: " << d[3]  << endl;
    return 0;
}
