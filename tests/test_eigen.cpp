#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <epgraph>
#include <iostream>
#include <vector>

void check(
    epg::Scalar& s,
    float expected_value,
    float expected_grad,
    bool expected_const = false) {
  bool check_value = std::fabs(s.get_value() - expected_value) < 1.0e-6;
  bool check_grad = std::fabs(s.get_grad() - expected_grad) < 1.0e-6;
  bool check_const = s.is_const() == expected_const;
  if (!check_value) {
    std::cout << "expected_value = " << expected_value << ", but got "
              << s.get_value() << std::endl;
  }
  if (!check_grad) {
    std::cout << "expected_grad = " << expected_grad << ", but got "
              << s.get_grad() << std::endl;
  }
  assert(check_value);
  assert(check_grad);
  assert(check_const);
}

typedef Eigen::Matrix<epg::Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix_epg;
typedef Eigen::Matrix<epg::Scalar, Eigen::Dynamic, 1> Vector_epg;
typedef Eigen::Map<Matrix_epg> epgMap;

void test_dot() {
  using namespace epg;
  int N = 4;
  std::vector<Scalar> x(N);
  std::vector<Scalar> y(N);

  Vector_epg x_eigen(N);
  Vector_epg y_eigen(N);
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f * i;
    y[i] = 3.0f * i;

    x_eigen[i] = 1.0f * i;
    y_eigen[i] = 3.0f * i;
  }

  Scalar dot = 0.0f;
  for (int i = 0; i < N; i++) {
    dot = dot + x[i] * y[i];
  }
  eval(dot);

  Scalar dot_eigen = x_eigen.transpose() * y_eigen;

  eval(dot_eigen);

  check(dot_eigen, dot.get_value(), 0.0f);
  for (int i = 0; i < N; i++) {
    check(x[i], 1.0f * i, 0.0f);
    check(y[i], 3.0f * i, 0.0f);
    check(x_eigen[i], 1.0f * i, 0.0f);
    check(y_eigen[i], 3.0f * i, 0.0f);
  }

  diff(dot);
  diff(dot_eigen);
  check(dot_eigen, dot.get_value(), 0.0f);
  for (int i = 0; i < N; i++) {
    check(x[i], 1.0f * i, y[i].get_value());
    check(y[i], 3.0f * i, x[i].get_value());
    check(x_eigen[i], 1.0f * i, y[i].get_value());
    check(y_eigen[i], 3.0f * i, x[i].get_value());
  }

  zero_grad(dot);
  zero_grad(dot_eigen);
  check(dot, dot.get_value(), 0.0f);
  for (int i = 0; i < N; i++) {
    check(x[i], 1.0f * i, 0.0f);
    check(y[i], 3.0f * i, 0.0f);
    check(x_eigen[i], 1.0f * i, 0.0f);
    check(y_eigen[i], 3.0f * i, 0.0f);
  }

  std::cout << "test_dot: ok\n";
}

int main() {
  test_dot();
}
