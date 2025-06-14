#pragma once

#include <core_variable.hpp>
#include <cmath>

namespace epg {

struct _Pow : public _Scalar {
  std::shared_ptr<_Scalar> var;
  float exponent;
  _Pow(const Scalar& input_var, const float input_exponent) {
    var = input_var.get_ptr();
    exponent = input_exponent;
  }
  void zero_grad() override {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() override {
    var->eval();
    value = std::pow(var->value, exponent);
  }
  void diff(const float seed) {
    var->diff(exponent * std::pow(var->value, exponent - 1.0f) * seed);
  }
};

Scalar pow(const Scalar x, const float exponent) {
  std::shared_ptr<_Scalar> var(new _Pow(x, exponent));
  return var;
}

} // namespace epg
