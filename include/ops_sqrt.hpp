#pragma once

#include <core_variable.hpp>
#include <cmath>

namespace epg {

struct _Sqrt : public _Scalar {
  std::shared_ptr<_Scalar> var;
  _Sqrt(const Scalar& input_var) {
    var = input_var.get_ptr();
  }
  void zero_grad() override {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() override {
    var->eval();
    value = std::sqrt(var->value);
  }
  void diff(const float seed) override {
    var->diff(0.5f / std::sqrt(var->value) * seed);
  }
};

Scalar sqrt(const Scalar& x) {
  std::shared_ptr<_Scalar> var(new _Sqrt(x));
  return var;
}

} // namespace epg
