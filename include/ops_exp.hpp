#pragma once

#include <core_variable.hpp>
#include <cmath>

namespace epg {

struct _Exp : public _Scalar {
  std::shared_ptr<_Scalar> var;
  _Exp(const Scalar& input_var) {
    var = input_var.get_ptr();
  }
  void zero_grad() override {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() override {
    var->eval();
    value = std::exp(var->value);
  }
  void diff(const float seed) override {
    var->diff(std::exp(var->value) * seed);
  }
};

Scalar exp(const Scalar& x) {
  std::shared_ptr<_Scalar> var = std::make_shared<_Exp>(x);
  return var;
}

} // namespace epg
