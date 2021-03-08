// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file defines the EmitC core ops

#ifndef EMITC_EMITC_CORE_OPS_H
#define EMITC_EMITC_CORE_OPS_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

#include "emitc_types.h"

namespace emitc {

/// Functions for unary elementwise ops
// AbsOp
// TODO support complex numbers
template <typename Src>
inline Src abs(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::abs);

  return unary<Src>(x, f);
}

// CeilOp
template <typename Src>
inline Src ceil(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::ceil);

  return unary<Src>(x, f);
}

// ExpOp
template <typename Src>
inline Src exp(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::exp);

  return unary<Src>(x, f);
}

// FloorOp
template <typename Src>
inline Src floor(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::floor);

  return unary<Src>(x, f);
}

// LogOp
template <typename Src>
inline Src log(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::log);

  return unary<Src>(x, f);
}

// SqrtOp
template <typename Src>
inline Src sqrt(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::sqrt);

  return unary<Src>(x, f);
}

// TanhOp
template <typename Src>
inline Src tanh(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::tanh);

  return unary<Src>(x, f);
}

/// Functions for binary elementwise ops.
// AddOp
template <typename Src>
inline Src add(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::plus<ET_Src>{};

  return binary<Src>(x, y, f);
}

// MulOp
template <typename Src>
inline Src mul(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::multiplies<ET_Src>{};

  return binary<Src>(x, y, f);
}

// SubOp
template <typename Src>
inline Src sub(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::minus<ET_Src>{};

  return binary<Src>(x, y, f);
}

/// Other ops
// BroadcastInDimOp
template <typename Dest, typename Src>
inline Dest
broadcast_in_dim(Src operand,
                 Tensor<int64_t, Src::rank()> broadcast_dimensions) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  std::vector<size_t> retainedDimensions(Dest::rank());
  std::iota(retainedDimensions.begin(), retainedDimensions.end(), 0);

  // Checks if broadcast_dimensions is a subset of 0 .. Dest::rank()
  retainedDimensions.erase(
      std::remove_if(retainedDimensions.begin(), retainedDimensions.end(),
                     [&broadcast_dimensions](size_t i) {
                       return std::find(broadcast_dimensions.begin(),
                                        broadcast_dimensions.end(),
                                        i) == broadcast_dimensions.end();
                     }),
      retainedDimensions.end());
  assert(retainedDimensions.size() == Src::rank());

  Dest result;
  for (size_t i = 0; i < result.size(); i++) {
    auto dest_index = result.unravel_index(i);

    // reverse mapping with broadcast_dimensions
    std::array<size_t, Src::rank()> src_index;
    for (size_t j = 0; j < src_index.size(); j++) {
      src_index[j] = dest_index[broadcast_dimensions(j)];
    }
    // Handle case of broadcasting dimensions of size 1
    for (size_t i = 0; i < src_index.size(); ++i) {
      if (Src::shape()[i] == 1) {
        src_index[i] = 0;
      }
    }

    result[i] = operand[operand.ravel_index(src_index)];
  }

  return result;
}

// DotOp
template <typename Dest, typename Lhs, typename Rhs>
Dest dot(Lhs lhs, Rhs rhs) {
  static_assert(is_tensor_of_dim<2, Lhs>::value, "Expected 2 dimensional lhs");
  static_assert(is_tensor_of_dim<2, Rhs>::value, "Expected 2 dimensional rhs");
  static_assert(Lhs::dim(1) == Rhs::dim(0),
                "Expected contracting dimension to match");
  Dest output;

  for (size_t m = 0; m < lhs.dim(0); m++) {
    for (size_t n = 0; n < lhs.dim(1); n++) {
      for (size_t k = 0; k < rhs.dim(1); k++) {
        output(m, k) += lhs(m, n) * rhs(n, k);
      }
    }
  }

  return output;
}

} // namespace emitc

#endif // EMITC_EMITC_CORE_OPS_H
