#include <epgraph>
#include <iostream>

// This example demonstrates usage of EPGraph for
// a non-linear algebraic equation:
//   exp(-x) * sin(5*x) - 0.5 = 0

int main(int argc, char *argv[]) {
  using namespace epg;

  // Main variable:
  Variable x = new_variable(0.0f);

  // Coefficients:
  Variable a = new_variable(-0.5f);
  Variable b = new_variable(-1.0f);
  Variable c = new_variable(5.0f);

  // Equation:
  Variable f = a + exp(b * x) * sin(c * x);

  for (int i = 0; i < 6; i++) {
    zero_grad(f);
    eval(f);
    diff(f);
    x->value = x->value - f->value / x->grad;
  }

  std::cout << "Solution:" << std::endl;
  std::cout << "x = " << x->value << std::endl;

  return 0;
}
