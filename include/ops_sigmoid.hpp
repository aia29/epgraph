#pragma once

#include <cmath>
#include <core_variable.hpp>

namespace epg {

Scalar sigmoid(const Scalar x) {
  return 1.0f / (1.0f + epg::exp(0.0f-x));
}

} // namespace epg
