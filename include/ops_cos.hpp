#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Cos : public _Variable {
  Variable var;
  _Cos(const Variable var_) { var = var_; }
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

Variable cos(const Variable x) {
  Variable var = std::make_shared<_Cos>(x);
  return var;
}

} // namespace epg
