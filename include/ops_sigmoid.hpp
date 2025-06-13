#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

float sigmoid(const float x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  } else {
    return std::exp(x) / (1.0f + std::exp(x));
  }
}

float sigmoid_prime(const float x) {
  return sigmoid(x)*(1.0f - sigmoid(x));
}

struct _Sigmoid : public _Scalar {
  std::shared_ptr<_Scalar> var;
  _Sigmoid(const Scalar &input_var) {
    var = input_var.get_ptr();
  }
  void zero_grad() override {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() override {
    var->eval();
    value = sigmoid(var->value);
  }
  void diff(const float seed) override {
    var->diff(sigmoid_prime(var->value) * seed);
  }
};

Scalar sigmoid(const Scalar &x) {
  std::shared_ptr<_Scalar> var(new _Sigmoid(x));
  return var;
}

} // namespace epg
