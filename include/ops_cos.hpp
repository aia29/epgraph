#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Cos : public _Scalar {
  Scalar var;
  _Cos(const Scalar var_) { var = var_; }
  void zero_grad() {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() {
    var->eval();
    value = std::cos(var->value);
  }
  void diff(const float seed) { var->diff(-std::sin(var->value) * seed); }
};

Scalar cos(const Scalar x) {
  Scalar var = std::make_shared<_Cos>(x);
  return var;
}

} // namespace epg
