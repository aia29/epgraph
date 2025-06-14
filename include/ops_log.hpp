#pragma once

#include <core_variable.hpp>
#include <cmath>

namespace epg {

struct _Log : public _Scalar {
  std::shared_ptr<_Scalar> var;
  _Log(const Scalar& input_var) {
    var = input_var.get_ptr();
  }
  void zero_grad() override {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() override {
    var->eval();
    value = std::log(var->value);
  }
  void diff(const float seed) override {
    var->diff(seed / var->value);
  }
};

Scalar log(const Scalar x) {
  std::shared_ptr<_Scalar> var(new _Log(x));
  return var;
}

} // namespace epg
