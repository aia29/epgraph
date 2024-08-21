#pragma once

#include <memory>

namespace epg {

struct _Scalar {
  float value;
  float grad;
  bool is_const;
  _Scalar(const bool is_const_ = false) {
    value = 0.0f;
    grad = 0.0f;
    is_const = is_const_;
  }
  _Scalar(const float value_, const bool is_const_ = false) {
    value = value_;
    grad = 0.0f;
    is_const = is_const_;
  }

  virtual void zero_grad() { grad = 0.0f; }
  virtual void eval() {}
  virtual void diff(const float seed) {
    if(is_const != true) {
      grad += seed;
    }
  }
};

typedef std::shared_ptr<_Scalar> Scalar;

Scalar new_const() {
  Scalar var = std::make_shared<_Scalar>(true);
  return var;
}

Scalar new_const(const float val) {
  Scalar var = std::make_shared<_Scalar>(val, true);
  return var;
}

Scalar make_variable() {
  Scalar var = std::make_shared<_Scalar>();
  return var;
}

Scalar make_variable(const float val) {
  Scalar var = std::make_shared<_Scalar>(val);
  return var;
}

} // namespace epg
