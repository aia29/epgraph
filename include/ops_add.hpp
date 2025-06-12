#pragma once
#include <core_variable.hpp>

namespace epg {

struct _Add : public _Scalar {
  std::shared_ptr<_Scalar> var1;
  std::shared_ptr<_Scalar> var2;
  _Add(const Scalar &input_var1, const Scalar &input_var2) {
    var1 = input_var1.get_ptr();
    var2 = input_var2.get_ptr();
  }
  void zero_grad() override {
    this->grad = 0.0f;
    var1->zero_grad();
    var2->zero_grad();
  }
  void eval() override {
    var1->eval();
    var2->eval();
    this->value = var1->value + var2->value;
  }
  void diff(const float input_seed) override {
    var1->diff(input_seed);
    var2->diff(input_seed);
  }
};

Scalar add(const Scalar &x, const Scalar &y) {
  std::shared_ptr<_Scalar> var(new _Add(x, y));
  return Scalar(var);
}

Scalar operator+(const Scalar &x, const Scalar &y) {
  return add(x, y);
}
Scalar operator+(const Scalar &x, const float y) {
  return add(x, Scalar(y, true));
}
Scalar operator+(const float x, const Scalar &y) {
  return add(Scalar(x, true), y);
}

Scalar& operator+=(Scalar &x, const Scalar &y) {
  x = x + y;
  return x;
}

} // namespace epg
