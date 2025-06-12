#pragma once

#include <memory>

namespace epg {

struct _Scalar {
  float value;
  float grad;
  bool is_const;
  _Scalar(const bool input_is_const = false) {
    value = 0.0f;
    grad = 0.0f;
    is_const = input_is_const;
  }
  _Scalar(const float input_value, const bool input_is_const = false) {
    value = input_value;
    grad = 0.0f;
    is_const = input_is_const;
  }
  virtual void zero_grad() { grad = 0.0f; }
  virtual void eval() {}
  virtual void diff(const float seed) {
    if(is_const != true) {
      grad += seed;
    }
  }
};


struct Scalar {
  Scalar(const float input_value = 0.0f, const bool input_is_const = false) {
    scalar = std::make_shared<_Scalar>(input_value, input_is_const);
  }
  Scalar(const std::shared_ptr<_Scalar> &input_other) {
    scalar = input_other;
  }
  void operator=(const std::shared_ptr<_Scalar> &input_other) {
    scalar = input_other;
  }
  void operator=(const Scalar &input_other) {
    scalar = input_other.scalar;
  }
  void operator=(const float input_value) {
    if(scalar) {
      scalar->value = input_value;
    } else {
      scalar = std::make_shared<_Scalar>(input_value);
    }
  }
  void zero_grad() const {
    scalar->zero_grad();
  }
  void eval() const {
    scalar->eval();
  }
  void diff(const float input_seed) const {
    scalar->diff(input_seed);
  }

  float get_grad() const {
    return scalar->grad;
  }

  float get_value() const {
    return scalar->value;
  }

  bool is_const() const {
    return scalar->is_const;
  }

  void set_value(const float input_val) const {
    scalar->value = input_val;
  }

  std::shared_ptr<_Scalar> get_ptr() const {
    return scalar;
  }

private:
  std::shared_ptr<_Scalar> scalar;
};

} // namespace epg
