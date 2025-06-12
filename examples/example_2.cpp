//////////////////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// a polynomial regression problem via least squares.
//
// Consider a third-order polynomial:
// f(x) = w0 + w1 * x + w2 * x * x + w3 * pow(x, 3).
// To approximate a function q(x) = sin(x), where x in [0, pi]
//
// The optimization problem would be:
// minimize(f(x, w) - q(x))^2 w.r.t w[0:4].
//
//////////////////////////////////////////////////////////////

#include <cstdlib>
#include <epgraph>
#include <iostream>

float randu(const float a, const float b) {
  return (((float)rand()) / (RAND_MAX + 1.0f)) * (b - a) + a;
}

int main(int argc, char* argv[]) {
  using namespace epg;

  // Function to approximate:
  Scalar q = 0.0f;

  // Weights:
  Scalar w0 = 0.1f;
  Scalar w1 = 0.1f;
  Scalar w2 = 0.1f;
  Scalar w3 = 0.1f;

  // Sample data:
  Scalar x = 0.0f;

  // Polynomial approximation function
  Scalar f = (w0 + w1 * x + w2 * pow(x, 2.0f) + w3 * pow(x, 3.0f));

  // Objective function:
  Scalar obj = (q - f) * (q - f);

  float alpha = 0.001f;

  for (int i = 0; i < 1000000; i++) {
    x = randu(0.0f, M_PI);
    q = sinf(x.get_value()) + randu(-0.01, 0.01);

    zero_grad(obj);
    eval(obj);
    diff(obj);

    w0 = w0.get_value() - alpha * w0.get_grad();
    w1 = w1.get_value() - alpha * w1.get_grad();
    w2 = w2.get_value() - alpha * w2.get_grad();
    w3 = w3.get_value() - alpha * w3.get_grad();
  }

  std::cout << "Weights:" << std::endl;
  std::cout << "w0 = " << w0.get_value() << std::endl;
  std::cout << "w1 = " << w1.get_value() << std::endl;
  std::cout << "w2 = " << w2.get_value() << std::endl;
  std::cout << "w3 = " << w3.get_value() << std::endl;

  std::cout << std::endl;
  std::cout << "Accuracy check:" << std::endl;
  for (int i = 0; i < 10; i++) {
    x = randu(0.0f, M_PI);
    eval(f);
    std::cout << std::endl;
    std::cout << "x = " << x.get_value() << std::endl;
    std::cout << "f(x) = " << f.get_value() << std::endl;
    std::cout << "sin(x) = " << sin(x.get_value()) << std::endl;
  }

  return 0;
}
