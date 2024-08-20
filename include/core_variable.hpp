#pragma once

#include <memory>

namespace epg {

struct _Variable {
  float value;
  float grad;
  bool is_const;
  _Variable(const bool is_const_ = false) {
    value = 0.0f;
    grad = 0.0f;
    is_const = is_const_;
  }
  _Variable(const float value_, const bool is_const_ = false) {
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

typedef std::shared_ptr<_Variable> Variable;

Variable new_const() {
  Variable var = std::make_shared<_Variable>(true);
  return var;
}

Variable new_const(const float val) {
  Variable var = std::make_shared<_Variable>(val, true);
  return var;
}

Variable new_variable() {
  Variable var = std::make_shared<_Variable>();
  return var;
}

Variable new_variable(const float val) {
  Variable var = std::make_shared<_Variable>(val);
  return var;
}

} // namespace epg
