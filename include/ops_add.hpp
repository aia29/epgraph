#pragma once

#include <core_variable.hpp>

namespace epg {

struct _Add : public _Scalar {
  Scalar var1;
  Scalar var2;
  _Add(const Scalar var1_, const Scalar var2_) {
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
    value = var1->value + var2->value;
  }
  void diff(const float seed) {
    var1->diff(seed);
    var2->diff(seed);
  }
};

Scalar add(const Scalar x, const Scalar y) {
  Scalar var = std::make_shared<_Add>(x, y);
  return var;
}

Scalar operator+(const Scalar x, const Scalar y) { return add(x, y); }
Scalar operator+(const Scalar x, const float y) { return add(x, make_const(y)); }
Scalar operator+(const float x, const Scalar y) { return add(make_const(x), y); }

} // namespace epg
