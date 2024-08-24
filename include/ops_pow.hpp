#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Pow : public _Scalar {
  Scalar var;
  float exponent;
  _Pow(const Scalar var_, const float exponent_) {
    var = var_;
    exponent = exponent_;
  }
  void zero_grad() {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() {
    var->eval();
    value = std::pow(var->value, exponent);
  }
  void diff(const float seed) {
    var->diff(exponent * std::pow(var->value, exponent - 1.0f) * seed);
  }
};

Scalar pow(const Scalar x, const float exponent) {
  Scalar var = std::make_shared<_Pow>(x, exponent);
  return var;
}

} // namespace epg
