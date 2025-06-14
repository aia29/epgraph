#pragma once

#include <core_variable.hpp>
#include <cmath>

namespace epg {

float tanh_prime(const float x) {
  return 1.0f - std::tanh(x) * std::tanh(x);
}

struct _Tanh : public _Scalar {
  std::shared_ptr<_Scalar> var;
  _Tanh(const Scalar& input_var) {
    var = input_var.get_ptr();
  }
  void zero_grad() override {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() override {
    var->eval();
    value = std::tanh(var->value);
  }
  void diff(const float seed) override {
    var->diff(tanh_prime(var->value) * seed);
  }
};

Scalar tanh(const Scalar& x) {
  std::shared_ptr<_Scalar> var(new _Tanh(x));
  return var;
}

} // namespace epg
