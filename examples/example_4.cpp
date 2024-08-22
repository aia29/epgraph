/////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// a non-linear system:
//   x1^2*x2^3 - x1*x2^3 - 1 = 0
//   x1^3 - x1*x2^3 - 4 = 0
// According to https://www.glynholton.com/solutions/exercise-solution-2-21/
// the solution is (1.74762, 0.91472)
//
/////////////////////////////////////////////////

#include <Eigen/Dense>
#include <epgraph>
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  using namespace epg;
  using Eigen::MatrixXf;
  using Eigen::VectorXf;

  Scalar x = make_variable(1.0f);
  Scalar y = make_variable(1.0f);

  // Equations:
  Scalar f0 = (x * x - x) * y * y * y - 1.0f;
  Scalar f1 = x * x * x - x * y * y * y - 4.0f;

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
    J_eigen(0, 0) = x->grad;
    J_eigen(0, 1) = y->grad;

    // Evaluate the second equation
    zero_grad(f1);
    eval(f1);
    diff(f1);

    // And populate its Jacobian
    J_eigen(1, 0) = x->grad;
    J_eigen(1, 1) = y->grad;

    // Populate vectors to do a Newton step
    F_eigen(0) = f0->value;
    F_eigen(1) = f1->value;

    x_eigen(0) = x->value;
    x_eigen(1) = y->value;

    x_eigen = x_eigen - J_eigen.inverse() * F_eigen;

    // Copy the intermediate solution back
    x->value = x_eigen(0);
    y->value = x_eigen(1);
  }

  std::cout << x->value << std::endl;
  std::cout << y->value << std::endl;

  return 0;
}
