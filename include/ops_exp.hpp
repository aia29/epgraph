#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Exp : public _Scalar {
  Scalar var;
  _Exp(const Scalar var_) { var = var_; }
  void zero_grad() {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() {
    var->eval();
    value = std::exp(var->value);
  }
  void diff(const float seed) { var->diff(std::exp(var->value) * seed); }
};

Scalar exp(const Scalar x) {
  Scalar var = std::make_shared<_Exp>(x);
  return var;
}

} // namespace epg
