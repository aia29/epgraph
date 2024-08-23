#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Div : public _Scalar {
  Scalar var1;
  Scalar var2;
  _Div(const Scalar var1_, const Scalar var2_) {
    var1 = var1_;
    var2 = var2_;
  }
  void zero_grad() {
    grad = 0.0f;
    var1->zero_grad();
    var2->zero_grad();
  }
  void eval() {
    var1->eval();
    var2->eval();
    value = var1->value / var2->value;
  }
  void diff(const float seed) {
    var1->diff(seed / var2->value);
    var2->diff( -seed * var1->value / (var2->value * var2->value));
  }
};

Scalar div(const Scalar x, const Scalar y) {
  Scalar var = std::make_shared<_Div>(x, y);
  return var;
}

Scalar operator/(const Scalar x, const Scalar y) { return div(x, y); }
Scalar operator/(const Scalar x, const float y) { return div(x, make_const(y)); }
Scalar operator/(const float x, const Scalar y) { return div(make_const(x), y); }

} // namespace epg
