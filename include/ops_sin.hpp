#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Sin : public _Variable {
  Variable var;
  _Sin(const Variable var_) { var = var_; }
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

Variable sin(const Variable x) {
  Variable var = std::make_shared<_Sin>(x);
  return var;
}

} // namespace epg
