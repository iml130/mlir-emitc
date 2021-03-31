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

// This file defines the EmitC core ops.

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

/// Functions for unary elementwise ops.
// AbsOp
// TODO: Add support for complex numbers.
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

// NegateOp
template <typename Src>
inline Src negate(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::negate<ET_Src>{};

  return unary<Src>(x, f);
}

// ReluNOp
template <typename Min, typename Src, typename Max>
inline Src clamp(Min min, Src operand, Max max) {
  static_assert(
      std::is_same<Min, Src>::value ||
          (is_tensor_of_dim<0, Min>::value &&
           std::is_same<typename get_element_type<Src>::type,
                        typename get_element_type<Min>::type>::value),
      "Expected the same type for min and operand or a 0-dim tensor of the "
      "same element type for min");
  static_assert(
      std::is_same<Max, Src>::value ||
          (is_tensor_of_dim<0, Max>::value &&
           std::is_same<typename get_element_type<Src>::type,
                        typename get_element_type<Max>::type>::value),
      "Expected the same type for min and operand or a 0-dim tensor of the "
      "same element type for max");

  const bool broadcast_min = !std::is_same<Min, Src>::value;
  const bool broadcast_max = !std::is_same<Max, Src>::value;

  Src result;
  for (size_t index = 0; index < Src::size(); index++) {
    const auto value_min = broadcast_min ? min[0] : min[index];
    const auto value_max = broadcast_max ? max[0] : max[index];

    auto value = operand[index];
    value = value < value_min ? value_min : value;
    value = value > value_max ? value_max : value;

    result[index] = value;
  }

  return result;
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

// MaxOp
template <typename Src>
inline Src max(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f =
      static_cast<const ET_Src &(*)(const ET_Src &, const ET_Src &)>(std::max);

  return binary<Src>(x, y, f);
}

// MinOp
template <typename Src>
inline Src min(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f =
      static_cast<const ET_Src &(*)(const ET_Src &, const ET_Src &)>(std::min);

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

/// Other ops.
// BroadcastInDimOp
// The broadcast_dimensions argument maps from Src to Dest dimensions
template <typename Dest, typename Src>
inline Dest
broadcast_in_dim(Src operand,
                 Tensor<int64_t, Src::rank()> broadcast_dimensions) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  std::vector<size_t> retainedDimensions(Dest::rank());
  std::iota(retainedDimensions.begin(), retainedDimensions.end(), 0);

  // Checks if broadcast_dimensions is a subset of 0 .. Dest::rank().
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

    // Reverse mapping with broadcast_dimensions
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

// ReshapeOp
template <typename Dest, typename Src>
inline Dest reshape(Src x) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Dest>::type;

  static_assert(std::is_same<ET_Src, ET_Dest>::value, "Element type mismatch");
  static_assert(Src::size() == Dest::size(), "Tensor size mismatch");

  Dest z;

  std::copy(x.begin(), x.end(), z.begin());

  return z;
}

// PadOp
// TODO: Add support for negative edge padding
template <typename Dest, typename Src>
inline Dest pad(Src operand,
                Tensor<typename get_element_type<Src>::type> padding_value,
                Tensor<int64_t, Src::rank()> edge_padding_low,
                Tensor<int64_t, Src::rank()> edge_padding_high,
                Tensor<int64_t, Src::rank()> interior_padding) {
  assert(std::all_of(interior_padding.begin(), interior_padding.end(),
                     [](int64_t i) { return i >= 0; }));

  assert(std::all_of(edge_padding_low.begin(), edge_padding_low.end(),
                     [](int64_t i) { return i >= 0; }));
  assert(std::all_of(edge_padding_high.begin(), edge_padding_high.end(),
                     [](int64_t i) { return i >= 0; }));

  Dest result;

  auto interior = [&interior_padding](std::array<size_t, Src::rank()> index) {
    for (size_t i = 0; i < index.size(); i++) {
      if (index[i] % (interior_padding[i] + 1) != 0) {
        return true;
      }
    }
    return false;
  };

  auto out_of_bounds = [](std::array<size_t, Src::rank()> index) {
    for (size_t i = 0; i < index.size(); i++) {
      if (index[i] < 0 || index[i] >= Src::dim(i)) {
        return true;
      }
    }
    return false;
  };

  for (size_t i = 0; i < result.size(); i++) {
    auto index = result.unravel_index(i);

    // Shift by low padding
    for (size_t j = 0; j < index.size(); j++) {
      index[j] -= edge_padding_low[j];
    }

    if (interior(index)) {
      result[i] = padding_value();
    } else {
      // Squeeze by interrior padding
      for (size_t j = 0; j < index.size(); j++) {
        size_t pad = interior_padding[j];
        assert(index[j] % (pad + 1) == 0);
        index[j] /= (pad + 1);
      }

      if (out_of_bounds(index)) {
        result[i] = padding_value();
      } else {
        result[i] = operand[operand.ravel_index(index)];
      }
    }
  }
  return result;
}

} // namespace emitc

#endif // EMITC_EMITC_CORE_OPS_H
