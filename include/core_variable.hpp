#pragma once

#include <memory>

namespace epg {

struct _Variable {
  float value;
  float grad;
  _Variable() {
    value = 0.0f;
    grad = 0.0f;
  }
  _Variable(const float value_) {
    value = value_;
    grad = 0.0f;
  }

  virtual void zero_grad() { grad = 0.0f; }
  virtual void eval() {}
  virtual void diff(const float seed) { grad += seed; }
};

typedef std::shared_ptr<_Variable> Variable;

} // namespace epg
