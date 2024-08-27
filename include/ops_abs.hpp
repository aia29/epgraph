#pragma once

#include <cassert>
#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Abs : public _Scalar {
  Scalar var;
  _Abs(const Scalar var_) { var = var_; }
  void zero_grad() {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() {
    var->eval();
    value = std::fabs(var->value);
  }
  void diff(const float seed) {
    assert(var->value != 0.0);
    var->diff(((var->value > 0.0) - (var->value < 0.0)) * seed);
  }
};

Scalar abs(const Scalar x) {
  Scalar var = std::make_shared<_Abs>(x);
  return var;
}

} // namespace epg
