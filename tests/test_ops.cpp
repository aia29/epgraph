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

void test_constructors() {
  using namespace epg;
  Scalar a(1.0f);
  Scalar b(1.0f, true);
  Scalar c = 3.0f;
  Scalar d = 3.0f;

  check(a, 1.0f, 0.0f);
  check(b, 1.0f, 0.0f, true);
  check(c, 3.0f, 0.0f);

  std::cout << "test_constructors: ok\n";
}

void test_add() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = 5.0f;
  Scalar z = x + y + y;

  zero_grad(z);
  eval(z);
  diff(z);
  check(x, 2.0f, 1.0f);
  check(y, 5.0f, 2.0f);
  check(z, 12.0f, 0.0f);

  std::cout << "test_add: ok\n";
}

void test_mul() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = 5.0f;
  Scalar z = x + y * y;

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, 1.0f);
  check(y, 5.0f, 10.0f);
  check(z, 27.0f, 0.0f);

  std::cout << "test_mul: ok\n";
}

void test_div() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = 5.0f;
  Scalar z = x / y;

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, 0.2);
  check(y, 5.0f, -0.08);
  check(z, 0.4f, 0.0f);

  std::cout << "test_div: ok\n";
}

void test_sin() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = 0.5f;
  Scalar z = x * sin(y);

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, std::sin(0.5));
  check(y, 0.5f, 2.0f * std::cos(0.5));
  check(z, 2.0f * std::sin(0.5), 0.0f);

  std::cout << "test_sin: ok\n";
}

void test_cos() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = 0.5f;
  Scalar z = x * cos(y);

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, std::cos(0.5));
  check(y, 0.5f, -2.0f * std::sin(0.5));
  check(z, 2.0f * std::cos(0.5), 0.0f);

  std::cout << "test_cos: ok\n";
}

void test_abs() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = -0.5f;
  Scalar z = x * abs(y);

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, 0.5f);
  check(y, -0.5f, -2.0f);
  check(z, 1.0f, 0.0f);

  std::cout << "test_abs: ok\n";
}

void test_exp() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = -0.5f;
  Scalar z = x * exp(y);

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, std::exp(-0.5f));
  check(y, -0.5f, 2.0f * std::exp(-0.5f));
  check(z, 2.0f * std::exp(-0.5f), 0.0f);

  std::cout << "test_exp: ok\n";
}

void test_pow() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = -0.5f;
  Scalar z = x * pow(y, 2.0f);

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, 0.25f);
  check(y, -0.5f, -2.0f);
  check(z, 0.5f, 0.0f);

  std::cout << "test_pow: ok\n";
}

void test_log() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = -0.5f;
  Scalar z = log(x) * y;

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, -0.25f);
  check(y, -0.5f, std::log(2.0f));
  check(z, -0.5 * std::log(2.0f), 0.0f);

  std::cout << "test_log: ok\n";
}

void test_sqrt() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = -0.5f;
  Scalar z = sqrt(x) * y;

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, -0.25f / std::sqrt(2.0f));
  check(y, -0.5f, std::sqrt(2.0f));
  check(z, -0.5 * std::sqrt(2.0f), 0.0f);

  std::cout << "test_sqrt: ok\n";
}

void test_tanh() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = -0.5f;
  Scalar z = tanh(x) * y;

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, -0.5 * tanh_prime(2.0f));
  check(y, -0.5f, std::tanh(2.0f));
  check(z, -0.5 * std::tanh(2.0f), 0.0f);

  std::cout << "test_tanh: ok\n";
}

void test_sigmoid() {
  using namespace epg;
  Scalar x = 2.0f;
  Scalar y = -0.5f;
  Scalar z = sigmoid(x) * y;

  zero_grad(z);
  eval(z);
  diff(z);

  check(x, 2.0f, -0.5 * sigmoid_prime(2.0f));
  check(y, -0.5f, sigmoid(2.0f));
  check(z, -0.5 * sigmoid(2.0f), 0.0f);

  std::cout << "test_sigmoid: ok\n";
}

void test_summation() {
  using namespace epg;
  Scalar x = 0.0;
  for (int i = 0; i < 10; i++) {
    x = x + Scalar(1.0f * i);
  }

  zero_grad(x);
  eval(x);
  diff(x);

  check(x, 45.0f, 0.0f);
  std::cout << "test_summation: ok\n";
}

void test_dot() {
  using namespace epg;
  int N = 4;
  std::vector<Scalar> x(N);
  std::vector<Scalar> y(N);
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f * i;
    y[i] = 3.0f * i;
  }

  Scalar dot = 0.0f;
  float dot_f = 0.0f;
  for (int i = 0; i < N; i++) {
    dot = dot + x[i] * y[i];
    dot_f = dot_f + x[i].get_value() * y[i].get_value();
  }

  eval(dot);

  check(dot, dot_f, 0.0f);
  for (int i = 0; i < N; i++) {
    check(x[i], 1.0f * i, 0.0f);
    check(y[i], 3.0f * i, 0.0f);
  }

  diff(dot);
  check(dot, dot_f, 0.0f);
  for (int i = 0; i < N; i++) {
    check(x[i], 1.0f * i, y[i].get_value());
    check(y[i], 3.0f * i, x[i].get_value());
  }

  zero_grad(dot);
  check(dot, dot_f, 0.0f);
  for (int i = 0; i < N; i++) {
    check(x[i], 1.0f * i, 0.0f);
    check(y[i], 3.0f * i, 0.0f);
  }

  std::cout << "test_dot: ok\n";
}

int main() {
  test_constructors();
  test_add();
  test_mul();
  test_div();
  test_sin();
  test_cos();
  test_abs();
  test_exp();
  test_pow();
  test_log();
  test_sqrt();
  test_tanh();
  test_sigmoid();
  test_summation();
  test_dot();
}
