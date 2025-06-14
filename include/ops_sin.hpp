#pragma once

#include <core_variable.hpp>
#include <cmath>

namespace epg {

struct _Sin : public _Scalar {
  std::shared_ptr<_Scalar> var;
  _Sin(const Scalar& input_var) {
    var = input_var.get_ptr();
  }
  void zero_grad() override {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() override {
    var->eval();
    value = std::sin(var->value);
  }
  void diff(const float seed) {
    var->diff(std::cos(var->value) * seed);
  }
};

Scalar sin(const Scalar& x) {
  std::shared_ptr<_Scalar> var(new _Sin(x));
  return var;
}

} // namespace epg
