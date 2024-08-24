#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Sqrt : public _Scalar {
  Scalar var;
  _Sqrt(const Scalar var_) { var = var_; }
  void zero_grad() {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() {
    var->eval();
    value = std::sqrt(var->value);
  }
  void diff(const float seed) {
    var->diff(0.5f / std::sqrt(var->value) * seed);
  }
};

Scalar sqrt(const Scalar x) {
  Scalar var = std::make_shared<_Sqrt>(x);
  return var;
}

} // namespace epg
