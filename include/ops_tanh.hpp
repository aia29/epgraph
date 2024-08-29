#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Tanh : public _Scalar {
  Scalar var;
  _Tanh(const Scalar var_) { var = var_; }
  void zero_grad() {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() {
    var->eval();
    value = std::tanh(var->value);
  }
  void diff(const float seed) { var->diff((1.0f - std::tanh(var->value)*std::tanh(var->value)) * seed); }
};

Scalar tanh(const Scalar x) {
  Scalar var = std::make_shared<_Tanh>(x);
  return var;
}

} // namespace epg
