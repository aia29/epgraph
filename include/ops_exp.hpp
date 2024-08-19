#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Exp : public _Variable {
  Variable var;
  _Exp(const Variable var_) { var = var_; }
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

Variable exp(const Variable x) {
  Variable var = std::make_shared<_Exp>(x);
  return var;
}

} // namespace epg
