#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Sin : public _Scalar {
  Scalar var;
  _Sin(const Scalar var_) { var = var_; }
  void zero_grad() {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() {
    var->eval();
    value = std::sin(var->value);
  }
  void diff(const float seed) { var->diff(std::cos(var->value) * seed); }
};

Scalar sin(const Scalar x) {
  Scalar var = std::make_shared<_Sin>(x);
  return var;
}

} // namespace epg
