#pragma once

#include <core_variable.hpp>

namespace epg {

struct _Sub : public _Scalar {
  Scalar var1;
  Scalar var2;
  _Sub(const Scalar var1_, const Scalar var2_) {
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
    value = var1->value - var2->value;
  }
  void diff(const float seed) {
    var1->diff(seed);
    var2->diff(-seed);
  }
};

Scalar sub(const Scalar x, const Scalar y) {
  Scalar var = std::make_shared<_Sub>(x, y);
  return var;
}

Scalar operator-(const Scalar x, const Scalar y) { return sub(x, y); }
Scalar operator-(const Scalar x, const float y) { return sub(x, make_const(y)); }
Scalar operator-(const float x, const Scalar y) { return sub(make_const(x), y); }

} // namespace epg
