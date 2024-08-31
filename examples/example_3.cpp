/////////////////////////////////////////////////
//
// This example demonstrates usage of EPGraph for
// a non-linear algebraic equation:
//   exp(-x) * sin(5*x) - 0.5 = 0
//
/////////////////////////////////////////////////

#include <epgraph>
#include <iostream>
#include <set>
#include <string>

float roundf(const float d) {
  return (round((int)(d * 10000.0))) / 10000.0;
}

int main(int argc, char* argv[]) {
  using namespace epg;

  // Main variable:
  Scalar x = make_variable();

  // Equation:
  Scalar f = exp(-1.0f * x) * sin(5.0f * x) - 0.5f;

  // Find its roots in range [xmin, xmax]
  std::set<float> roots;
  float xmin = -4.0;
  float xmax = 1.0;
  for (float xi = xmin; xi <= xmax; xi = xi + 0.1) {
    x->value = xi;
    for (int i = 0; i < 8; i++) {
      zero_grad(f);
      eval(f);
      diff(f);
      x->value = x->value - f->value / x->grad;
    }
    if (x->value >= xmin && x->value <= xmax) {
      roots.insert(roundf(x->value));
    }
  }

  std::cout << "Numerical solution:" << std::endl;
  for (const float root : roots) {
    std::cout << root << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Roots from WolframAlpha:" << std::endl;
  std::cout << " -3.76760038832412 " << std::endl;
  std::cout << " -3.14589582169776 " << std::endl;
  std::cout << " -2.50510515065551 " << std::endl;
  std::cout << " -1.89992751799026 " << std::endl;
  std::cout << " -1.22722049943874 " << std::endl;
  std::cout << " -0.679561266576494 " << std::endl;
  std::cout << "  0.119749199036199 " << std::endl;
  std::cout << "  0.448443973308867 " << std::endl;

  return 0;
}
