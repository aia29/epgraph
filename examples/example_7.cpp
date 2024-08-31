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

void solve_tridiagonal(const int X, float *x,
                       const float *a, const float *b,
                       const float *c, float *scratch) {
    scratch[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    for (int ix = 1; ix < X; ix++) {
        if (ix < X-1){
        scratch[ix] = c[ix] / (b[ix] - a[ix] * scratch[ix - 1]);
        }
        x[ix] = (x[ix] - a[ix] * x[ix - 1]) / (b[ix] - a[ix] * scratch[ix - 1]);
    }

    for (int ix = X - 2; ix >= 0; ix--)
        x[ix] -= scratch[ix] * x[ix + 1];
}


void solve_step(std::vector<epg::Scalar> &u, std::vector<epg::Scalar> &uold,
                const float mu, const float dt, const float dx) {
  using Eigen::MatrixXf;
  using Eigen::VectorXf;

  const int N = u.size() - 2;

  VectorXf a_eigen = VectorXf::Zero(N);
  VectorXf b_eigen = VectorXf::Zero(N);
  VectorXf c_eigen = VectorXf::Zero(N);
  VectorXf scratch_eigen = VectorXf::Zero(N);

  VectorXf u_eigen(N);
  VectorXf F_eigen(N);

  const float dx_inv = 1.0f / dx;
  const float dx2_inv = 1.0f / dx / dx;

  for (int iter = 0; iter < 8; iter++) {
    for (int i = 1; i < u.size() - 1; i++) {

      // modpoint values
      epg::Scalar up = 0.5f*(u[i + 1] + uold[i + 1]->value);
      epg::Scalar uc = 0.5f*(u[i] + uold[i]->value);
      epg::Scalar um = 0.5f*(u[i - 1] + uold[i - 1]->value);

      epg::Scalar EQ =
          (u[i] - uold[i]->value) / dt
          + uc * 0.5 * (up - um) * dx_inv
          - mu * (up - 2.0 * uc + um) * dx2_inv;

      zero_grad(EQ);
      eval(EQ);
      diff(EQ);

      // And populate its tridiagonal Jacobian
      if (i > 1) {
        a_eigen(i - 1) = u[i - 1]->grad;
      }
      b_eigen(i - 1) = u[i]->grad;
      if (i < N) {
        c_eigen(i) = u[i + 1]->grad;
      }
      F_eigen(i - 1) = EQ->value;
      u_eigen(i - 1) = u[i]->value;
    }

    solve_tridiagonal(N, F_eigen.data(),
                      a_eigen.data(), b_eigen.data(),
                      c_eigen.data(), scratch_eigen.data());

    u_eigen = u_eigen - F_eigen;
  }
  for (int i = 1; i < u.size() - 1; i++) {
    u[i]->value = u_eigen(i - 1);
  }
}

int main(int argc, char *argv[]) {

  const int Nt = 32;
  const int Nx = 1001;
  const int save_every = 8;
  const float dx = 2.0f * M_PI / (Nx - 1.0f);
  const float dt = 0.25f;
  float mu = 0.025f;

  std::cout << "dt = " << dt << std::endl;
  std::cout << "dx = " << dx << std::endl;
  std::cout << "mu = " << mu << std::endl;
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
