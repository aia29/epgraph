//////////////////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// a polynomial regression problem via least squares.
//
// Consider a third-order polynomial:
// f(x) = w0 + w1 * x + w2 * x * x + w3 * x * x * x.
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

int main(int argc, char *argv[]) {
  using namespace epg;

  // Function to approximate:
  Scalar q = make_variable(0.0f);

  // Weights:
  Scalar w0 = make_variable(0.1f);
  Scalar w1 = make_variable(0.1f);
  Scalar w2 = make_variable(0.1f);
  Scalar w3 = make_variable(0.1f);

  // Sample data:
  Scalar x = make_variable(0.0f);

  // Polynomial approximation function
  Scalar f = (w0 + w1 * x + w2 * x * x + w3 * x * x * x);

  // Objective function:
  Scalar obj = (q - f) * (q - f);

  float alpha = 0.001f;

  for (int i = 0; i < 1000000; i++) {
    x->value = randu(0.0f, M_PI);
    q->value = sinf(x->value) + randu(-0.01, 0.01);

    zero_grad(obj);
    eval(obj);
    diff(obj);

    w0->value = w0->value - alpha * w0->grad;
    w1->value = w1->value - alpha * w1->grad;
    w2->value = w2->value - alpha * w2->grad;
    w3->value = w3->value - alpha * w3->grad;
  }

  std::cout << "Weights:" << std::endl;
  std::cout << "w0 = " << w0->value << std::endl;
  std::cout << "w1 = " << w1->value << std::endl;
  std::cout << "w2 = " << w2->value << std::endl;
  std::cout << "w3 = " << w3->value << std::endl;

  std::cout << std::endl;
  std::cout << "Accuracy check:" << std::endl;
  for (int i = 0; i < 10; i++) {
    x->value = randu(0.0f, M_PI);
    eval(f);
    std::cout << std::endl;
    std::cout << "x = " << x->value << std::endl;
    std::cout << "f(x) = " << f->value << std::endl;
    std::cout << "sin(x) = " << sin(x->value) << std::endl;
  }

  return 0;
}
