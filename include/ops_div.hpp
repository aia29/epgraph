#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Div : public _Scalar {
  std::shared_ptr<_Scalar> var1;
  std::shared_ptr<_Scalar> var2;
  _Div(const Scalar &input_var1, const Scalar &input_var2) {
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
    value = var1->value / var2->value;
  }
  void diff(const float seed) override {
    var1->diff(seed / var2->value);
    var2->diff(-seed * var1->value / (var2->value * var2->value));
  }
};

Scalar div(const Scalar &x, const Scalar &y) {
  std::shared_ptr<_Scalar> var(new _Div(x, y));
  return var;
}

Scalar operator/(const Scalar &x, const Scalar &y) { return div(x, y); }
Scalar operator/(const Scalar &x, const float y) {
  return div(x, Scalar(y, true));
}
Scalar operator/(const float x, const Scalar &y) {
  return div(Scalar(x, true), y);
}

} // namespace epg
