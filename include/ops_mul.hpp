#pragma once

#include <core_variable.hpp>
#include <cmath>

namespace epg {

struct _Mul : public _Scalar {
  std::shared_ptr<_Scalar> var1;
  std::shared_ptr<_Scalar> var2;
  _Mul(const Scalar& input_var1, const Scalar& input_var2) {
    var1 = input_var1.get_ptr();
    var2 = input_var2.get_ptr();
  }
  void zero_grad() override {
    grad = 0.0f;
    var1->zero_grad();
    var2->zero_grad();
  }
  void eval() override {
    var1->eval();
    var2->eval();
    value = var1->value * var2->value;
  }
  void diff(const float input_seed) override {
    var1->diff(var2->value * input_seed);
    var2->diff(var1->value * input_seed);
  }
};

Scalar mul(const Scalar& x, const Scalar& y) {
  std::shared_ptr<_Scalar> var = std::make_shared<_Mul>(x, y);
  return var;
}

Scalar operator*(const Scalar& x, const Scalar& y) {
  return mul(x, y);
}
Scalar operator*(const Scalar& x, const float y) {
  return mul(x, Scalar(y, true));
}
Scalar operator*(const float x, const Scalar& y) {
  return mul(Scalar(x, true), y);
}

} // namespace epg
