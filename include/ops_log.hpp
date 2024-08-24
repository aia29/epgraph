#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

struct _Log : public _Scalar {
  Scalar var;
  _Log(const Scalar var_) { var = var_; }
  void zero_grad() {
    grad = 0.0f;
    var->zero_grad();
  }
  void eval() {
    var->eval();
    value = std::log(var->value);
  }
  void diff(const float seed) { var->diff(seed / var->value); }
};

Scalar log(const Scalar x) {
  Scalar var = std::make_shared<_Log>(x);
  return var;
}

} // namespace epg
