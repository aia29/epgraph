#include <epgraph>
#include <iostream>

int main(int argc, char* argv[]) {
  using namespace epg;

  Scalar y = 2.0f;
  Scalar x = 3.0f;
  Scalar z = 4.0f;
  Scalar f = x * (x + y) + y * y - (x / y) + z * sqrt(z) - exp(log(z));

  zero_grad(f);
  eval(f);
  diff(f);

  std::cout << "f = x * (x + y) + y * y - (x / y) + z * sqrt(z) - exp(log(z))"
            << std::endl;
  std::cout << "f(" << x.get_value() << ", " << y.get_value() << ", " << z.get_value()
            << ") = " << f.get_value() << std::endl;
  std::cout << "∂f/∂x = " << x.get_grad() << std::endl;
  std::cout << "∂f/∂y = " << y.get_grad() << std::endl;
  std::cout << "∂f/∂z = " << z.get_grad() << std::endl;

  Scalar g = sin(x) + x * y;
  x = M_PI;
  y = 2.0f;

  zero_grad(g);
  eval(g);
  diff(g);

  std::cout << std::endl;
  std::cout << "g = sin(x) + x * y" << std::endl;
  std::cout << "g(" << x.get_value() << ", " << y.get_value() << ") = " << g.get_value()
            << std::endl;
  std::cout << "∂g/∂x = " << x.get_grad() << std::endl;
  std::cout << "∂g/∂y = " << y.get_grad() << std::endl;

  Scalar q = abs(x) + abs(y);
  x = -M_PI;
  y = 2.0f;
  zero_grad(q);
  eval(q);
  diff(q);

  std::cout << std::endl;
  std::cout << "q = abs(x) + abs(y)" << std::endl;
  std::cout << "q(" << x.get_value() << ", " << y.get_value() << ") = " << q.get_value()
            << std::endl;
  std::cout << "∂q/∂x = " << x.get_grad() << std::endl;
  std::cout << "∂q/∂y = " << y.get_grad() << std::endl;

  return 0;
}
