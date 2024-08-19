#pragma once

#include <core_variable.hpp>

namespace epg {

struct _Sub : public _Variable {
  Variable var1;
  Variable var2;
  _Sub(const Variable var1_, const Variable var2_) {
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

Variable sub(const Variable x, const Variable y) {
  Variable var = std::make_shared<_Sub>(x, y);
  return var;
}

Variable operator-(const Variable x, const Variable y) { return sub(x, y); }

} // namespace epg
