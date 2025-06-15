#pragma once

#include <core_variable.hpp>
#include <cassert>
#include <cmath>

namespace epg {

struct _Abs : public _Scalar {
  std::shared_ptr<_Scalar> var;
  _Abs(const Scalar& input_var) {
    var = input_var.get_ptr();
  }
  void zero_grad() override {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() override {
    var->eval();
    value = std::fabs(var->value);
  }
  void diff(const float seed) {
    assert(var->value != 0.0);
    var->diff(((var->value > 0.0) - (var->value < 0.0)) * seed);
  }
};

Scalar abs(const Scalar& x) {
  std::shared_ptr<_Scalar> var = std::make_shared<_Abs>(x);
  return var;
}

} // namespace epg
