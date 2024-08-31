/////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// a Van der Pol equation:
//   d^2x/dx^2 - mu*(1 -x^2)*dx/dt - x = 0
// using an implicit midpoint method.
//
/////////////////////////////////////////////////

#include <Eigen/Dense>
#include <epgraph>
#include <fstream>

int main(int argc, char *argv[]) {
  using namespace epg;
  using Eigen::MatrixXf;
  using Eigen::VectorXf;

  const float dt = 0.25f;
  const float mu = 3.125f;

  float t0 = 0.0f;
  float x0 = 0.1f;
  float y0 = 0.1f;

  // Temporary Eigen variables
  MatrixXf J_eigen(2, 2);
  VectorXf x_eigen(2);
  VectorXf F_eigen(2);

  const int N = std::round(25.0f / dt);

  std::ofstream file("van_der_pol.csv");
  file << "t,x,y" << std::endl;
  file << t0 << "," << x0 << "," << y0 << std::endl;

  for (int n = 0; n < N; n++) {
    Scalar x = make_variable(x0);
    Scalar y = make_variable(y0);

    // modpoint values
    Scalar xm = 0.5f*(x + x0);
    Scalar ym = 0.5f*(y + y0);

    Scalar F0 = (x - x0) - dt * ym;
    Scalar F1 = (y - y0) - dt * (mu * (1.0f - xm * xm) * ym - xm);

    for (int i = 0; i < 8; i++) {
      // Evaluate the first equation
      zero_grad(F0);
      eval(F0);
      diff(F0);

      // And populate its Jacobian
      J_eigen(0, 0) = x->grad;
      J_eigen(0, 1) = y->grad;

      // Evaluate the second equation
      zero_grad(F1);
      eval(F1);
      diff(F1);

      // And populate its Jacobian
      J_eigen(1, 0) = x->grad;
      J_eigen(1, 1) = y->grad;

      // Populate vectors to do a Newton step
      F_eigen(0) = F0->value;
      F_eigen(1) = F1->value;

      x_eigen(0) = x->value;
      x_eigen(1) = y->value;

      x_eigen = x_eigen - J_eigen.inverse() * F_eigen;

      // Copy the intermediate solution back
      x->value = x_eigen(0);
      y->value = x_eigen(1);
    }

    x0 = x->value;
    y0 = y->value;
    t0 = t0 + dt;

    file << t0 << "," << x0 << "," << y0 << std::endl;
  }

  file.close();

  return 0;
}
