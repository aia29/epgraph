#include <epgraph>
#include <iostream>

int main(int argc, char *argv[]) {
  using namespace epg;

  Scalar y = new_variable(2.0);
  Scalar x = new_variable(3.0);
  Scalar z = x * (x + y) + y * y - (x / y);

  zero_grad(z);
  eval(z);
  diff(z);

  std::cout << "z = " << z->value << std::endl;
  std::cout << "∂z/∂x = " << x->grad << ", "
            << "∂z/∂y = " << y->grad << std::endl;

  return 0;
}
