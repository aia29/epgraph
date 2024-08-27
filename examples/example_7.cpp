/////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// a Burgers' equation:
//   du/dt + u * du/dx = mu * d2u/dx2
// using an implicit midpoint method.
//
/////////////////////////////////////////////////

#include <Eigen/Dense>
#include <epgraph>
#include <fstream>
#include <iostream>

void solve_step(std::vector<epg::Scalar> &u, std::vector<epg::Scalar> &uold,
                const float mu, const float dt, const float dx) {
  using Eigen::MatrixXf;
  using Eigen::VectorXf;

  const int N = u.size() - 2;

  MatrixXf J_eigen(N, N);
  VectorXf u_eigen(N);
  VectorXf F_eigen(N);
  VectorXf Delta_eigen(N);

  const float dx_inv = 1.0f / dx;
  const float dx2_inv = 1.0f / dx / dx;

  for (int iter = 0; iter < 8; iter++) {
    for (int i = 1; i < u.size() - 1; i++) {
      epg::Scalar EQ =
          (u[i] - uold[i]->value) / dt +
          0.5 * (u[i] + uold[i]->value) * 0.5 *
              (0.5 * (u[i + 1] - u[i - 1]) * dx_inv +
               0.5 * (uold[i + 1]->value - uold[i - 1]->value) * dx_inv) -
          mu * (0.5 * (u[i + 1] - 2.0 * u[i] + u[i - 1]) * dx2_inv +
                0.5 *
                    (uold[i + 1]->value - 2.0 * uold[i]->value +
                     uold[i - 1]->value) *
                    dx2_inv);

      zero_grad(EQ);
      eval(EQ);
      diff(EQ);

      // And populate its Jacobian
      if (i > 1) {
        J_eigen(i - 1, i - 2) = u[i - 1]->grad;
      }
      J_eigen(i - 1, i - 1) = u[i]->grad;
      if (i < N) {
        J_eigen(i - 1, i) = u[i + 1]->grad;
      }
      F_eigen(i - 1) = EQ->value;
      u_eigen(i - 1) = u[i]->value;
    }

    Delta_eigen = J_eigen.completeOrthogonalDecomposition().solve(F_eigen);
    u_eigen = u_eigen - Delta_eigen;
  }
  for (int i = 1; i < u.size() - 1; i++) {
    u[i]->value = u_eigen(i - 1);
  }
}

int main(int argc, char *argv[]) {

  const int Nt = 32;
  const int Nx = 101;
  const int save_every = 8;
  const float dx = 2.0f * M_PI / (Nx - 1.0f);
  const float dt = 0.25f;
  float mu = 0.0625f;

  std::cout << "dt = " << dt << std::endl;
  std::cout << "dx = " << dx << std::endl;
  std::cout << "mu dt/dx^2 = " << mu * dt / dx / dx << std::endl;

  std::vector<epg::Scalar> u(Nx);
  std::vector<epg::Scalar> uold(Nx);

  for (int i = 0; i < Nx; i++) {
    const float x = i * dx;
    u[i] = epg::make_variable(std::sin(x));
    uold[i] = epg::make_variable(std::sin(x));
  }

  std::ofstream file("burgers.csv");
  file << "x,u" << std::endl;

  for (int n = 0; n < Nt / 2; n++) {
    solve_step(u, uold, mu, dt, dx);
    solve_step(uold, u, mu, dt, dx);

    if (n % (save_every / 2) == 0) {

      for (int i = 0; i < Nx; i++) {
        const float x = i * dx;
        file << x << "," << u[i]->value << std::endl;
      }
    }
  }

  file.close();

  return 0;
}
