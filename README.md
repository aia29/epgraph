## EPGraph: An easy-peasy automatic differentiation library based on reverse-mode accumulation of computational graph.

### NOTE: EPGraph is under development. Functionality can be limited and things may change over time.

EPGraph can be used for variety of projects including but not limited to machine learning, ordinary and partial differential equations, non-linear equations and optimization. Due to header-only nature of the library, it is easy-peasy to use EPGraph:

```C++
#include <epgraph>
#include <iostream>

int main(int argc, char *argv[]) {
  using namespace epg;

  Scalar y = 2.0;
  Scalar x = 3.0;
  Scalar z = x * (x + y) + y * y;

  eval(z);
  diff(z);

  std::cout << "z = " << z.get_value() << std::endl;
  std::cout << "∂z/∂x = " << x.get_grad() << ", "
            << "∂z/∂y = " << y.get_grad() << std::endl;

  return 0;
}
```
