#include <epgraph>
#include <iostream>

int main(int argc, char* argv[]) {
  using namespace epg;

  Scalar y = make_variable(2.0f);
  Scalar x = make_variable(3.0f);
  Scalar z = make_variable(4.0f);
  Scalar f = x * (x + y) + y * y - (x / y) + z * sqrt(z) - exp(log(z));

  zero_grad(f);
  eval(f);
  diff(f);

  std::cout << "f = x * (x + y) + y * y - (x / y) + z * sqrt(z) - exp(log(z))"
            << std::endl;
  std::cout << "f(" << x->value << ", " << y->value << ", " << z->value
            << ") = " << f->value << std::endl;
  std::cout << "∂f/∂x = " << x->grad << std::endl;
  std::cout << "∂f/∂y = " << y->grad << std::endl;
  std::cout << "∂f/∂z = " << z->grad << std::endl;

  Scalar g = sin(x) + x * y;
  x->value = M_PI;
  y->value = 2.0f;

  zero_grad(g);
  eval(g);
  diff(g);

  std::cout << std::endl;
  std::cout << "g = sin(x) + x * y" << std::endl;
  std::cout << "g(" << x->value << ", " << y->value << ") = " << g->value
            << std::endl;
  std::cout << "∂g/∂x = " << x->grad << std::endl;
  std::cout << "∂g/∂y = " << y->grad << std::endl;

  Scalar q = abs(x) + abs(y);
  x->value = -M_PI;
  y->value = 2.0f;
  zero_grad(q);
  eval(q);
  diff(q);

  std::cout << std::endl;
  std::cout << "q = abs(x) + abs(y)" << std::endl;
  std::cout << "q(" << x->value << ", " << y->value << ") = " << q->value
            << std::endl;
  std::cout << "∂q/∂x = " << x->grad << std::endl;
  std::cout << "∂q/∂y = " << y->grad << std::endl;

  return 0;
}
