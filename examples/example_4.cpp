/////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// a non-linear system:
//   exp(-exp(-(x+y))) = y*(1+x^2)
//   x*cos(y) + y*sin(x) = 0.5
// According to https://www.mathworks.com/help/optim/ug/fsolve.html
// the solution is (0.3532    0.6061)
//
/////////////////////////////////////////////////

#include <Eigen/Dense>
#include <epgraph>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  using namespace epg;
  using Eigen::MatrixXf;
  using Eigen::VectorXf;

  Scalar x = 1.0f;
  Scalar y = 1.0f;

  Scalar f0 = exp(-exp(-(x + y))) - y * (1.0f + x * x);
  Scalar f1 = x * cos(y) + y * sin(x) - 0.5f;

  // Temporary Eigen variables
  MatrixXf J_eigen(2, 2);
  VectorXf x_eigen(2);
  VectorXf F_eigen(2);

  for (int i = 0; i < 8; i++) {
    // Evaluate the first equation
    zero_grad(f0);
    eval(f0);
    diff(f0);

    // And populate its Jacobian
    J_eigen(0, 0) = x.get_grad();
    J_eigen(0, 1) = y.get_grad();

    // Evaluate the second equation
    zero_grad(f1);
    eval(f1);
    diff(f1);

    // And populate its Jacobian
    J_eigen(1, 0) = x.get_grad();
    J_eigen(1, 1) = y.get_grad();

    // Populate vectors to do a Newton step
    F_eigen(0) = f0.get_value();
    F_eigen(1) = f1.get_value();

    x_eigen(0) = x.get_value();
    x_eigen(1) = y.get_value();

    // Eigen's inverse for 2x2 matrix is efficient
    // enough to be usef here as is:
    x_eigen = x_eigen - J_eigen.inverse() * F_eigen;

    // Copy the intermediate solution back
    x = x_eigen(0);
    y = x_eigen(1);
  }
  std::cout << "Roots calculated with the help of EPGraph:" << std::endl;
  std::cout << x.get_value() << ", " << y.get_value() << std::endl;
  std::cout << "Roots calculated with the help of Matlab:" << std::endl;
  std::cout << "0.3532, 0.6061" << std::endl;

  return 0;
}
