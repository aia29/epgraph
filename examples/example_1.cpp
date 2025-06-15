///////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// calculation of sqrt(2) using Newton’s iterations
// for f(x) = x*x – 2.
//
///////////////////////////////////////////////////

#include <epgraph>
#include <iostream>

int main(int argc, char* argv[]) {
  using namespace epg;

  Scalar x = 3.0f;
  Scalar f = x * x - 2.0f;

  for (int i = 0; i < 6; i++) {
    zero_grad(f);
    eval(f);
    diff(f);
    x = x.get_value() - f.get_value() / x.get_grad();
  }

  std::cout << "x = " << x.get_value() << std::endl;
  std::cout << "sqrt(2) = " << sqrtf(2.0f) << std::endl;

  return 0;
}
