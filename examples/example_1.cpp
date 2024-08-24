///////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// calculation of sqrt(2) using Newton’s iterations
// for f(x) = x*x – 2.
///////////////////////////////////////////////////

#include <epgraph>
#include <iostream>

int main(int argc, char *argv[]) {
  using namespace epg;

  Scalar x = make_variable(3.0f);
  Scalar y = make_variable(2.0f);
  Scalar f = x * x - y;

  for (int i = 0; i < 6; i++) {
    zero_grad(f);
    eval(f);
    diff(f);
    x->value = x->value - f->value / x->grad;
  }

  std::cout << "x = " << x->value << std::endl;
  std::cout << "sqrt(2) = " << sqrtf(2.0f) << std::endl;

  return 0;
}
