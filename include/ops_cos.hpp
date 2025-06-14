#pragma once

#include <core_variable.hpp>
#include <cmath>

namespace epg {

struct _Cos : public _Scalar {
  std::shared_ptr<_Scalar> var;
  _Cos(const Scalar& input_var) {
    var = input_var.get_ptr();
  }
  void zero_grad() override {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() override {
    var->eval();
    value = std::cos(var->value);
  }
  void diff(const float seed) {
    var->diff(-std::sin(var->value) * seed);
  }
};

Scalar cos(const Scalar& x) {
  std::shared_ptr<_Scalar> var(new _Cos(x));
  return var;
}

} // namespace epg
