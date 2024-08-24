#include <epgraph>
#include <iostream>

int main(int argc, char *argv[]) {
  using namespace epg;

  Scalar y = make_variable(2.0f);
  Scalar x = make_variable(3.0f);
  Scalar z = make_variable(4.0f);
  Scalar f = x * (x + y) + y * y - (x / y) + z * sqrt(z) - exp(log(z));

  zero_grad(f);
  eval(f);
  diff(f);

  std::cout << "f = " << z->value << std::endl;
  std::cout << "∂f/∂x = " << x->grad << std::endl;
  std::cout << "∂f/∂y = " << y->grad << std::endl;
  std::cout << "∂z/∂y = " << z->grad << std::endl;

  return 0;
}
