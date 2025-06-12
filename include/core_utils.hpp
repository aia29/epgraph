#pragma once

#include <cmath>
#include <ostream>

#include <core_variable.hpp>

std::ostream& operator<< (std::ostream& stream, const epg::Scalar& s) {
    stream << s.get_value() << ", " << s.get_grad();
    return stream;
}

bool operator==(const epg::Scalar& a, const epg::Scalar& b) {
  bool check_value = std::fabs(a.get_value() - b.get_value())< 1.0e-6;
  bool check_grad = std::fabs(a.get_grad() - b.get_grad()) < 1.0e-6;
  return check_value && check_grad;
}

bool operator==(const epg::Scalar& a, const float b) {
  bool check_value = std::fabs(a.get_value() - b)< 1.0e-6;
  return check_value;
}

bool operator==(const float b, const epg::Scalar& a) {
  bool check_value = std::fabs(a.get_value() - b)< 1.0e-6;
  return check_value;
}

namespace epg {

void eval(const Scalar& var) { var.eval(); }

void diff(const Scalar& var) { var.diff(1); }

void zero_grad(const Scalar& var) { var.zero_grad(); }

} // namespace epg
