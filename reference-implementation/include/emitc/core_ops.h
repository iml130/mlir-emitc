// Copyright Fraunhofer-Gesellschaft zur Förderung der angewandten
//           Forschung e.V.
//
// Licensed under the Apache License, Version 2.0  with LLVM exceptions (the
// "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file defines the EmitC core ops.

#ifndef EMITC_CORE_OPS_H
#define EMITC_CORE_OPS_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

#include "emitc/types.h"

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

// ConvertOp
template <typename Dest, typename Src>
inline Dest convert(Src x) {
  using ET_Dest = typename get_element_type<Dest>::type;
  using ET_Src = typename get_element_type<Src>::type;

  auto cast = [](ET_Src value) { return static_cast<ET_Dest>(value); };

  return unary<Dest, Src, UnaryFuncType<ET_Dest, ET_Src>>(x, cast);
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

// PowOp
template <typename Src>
inline Src pow(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = [](ET_Src a, ET_Src b) -> ET_Src {
    if (std::is_integral<ET_Src>::value) {
      const bool negative = b < 0;
      if (b < 0) {
        b = -b;
      }

      ET_Src result = 1;

      for (ET_Src i = 0; i < b; i++) {
        result *= a;
      }

      if (negative) {
        result = 1 / result;
      }
      return result;
    } else {
      return std::pow(a, b);
    }
  };

  return binary<Src>(x, y, f);
}

// SubtractOp
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

// BatchMatmulOp
template <typename Dest, typename Lhs, typename Rhs>
Dest batch_matmul(Lhs lhs, Rhs rhs) {
  static_assert(is_tensor_of_dim<3, Lhs>::value, "Expected 3 dimensional lhs");
  static_assert(is_tensor_of_dim<3, Rhs>::value, "Expected 3 dimensional rhs");
  static_assert(Lhs::dim(0) == Rhs::dim(0) && Lhs::dim(0) == Dest::dim(0),
                "Expected batch dimension to match");
  static_assert(Lhs::dim(2) == Rhs::dim(1),
                "Expected contracting dimension to match");
  static_assert(Dest::dim(1) == Lhs::dim(1), "Expected row dimension to match");
  static_assert(Dest::dim(2) == Rhs::dim(2),
                "Expected column dimension to match");
  Dest output;

  for (size_t b = 0; b < lhs.dim(0); b++) {
    for (size_t m = 0; m < lhs.dim(1); m++) {
      for (size_t n = 0; n < lhs.dim(2); n++) {
        for (size_t k = 0; k < rhs.dim(2); k++) {
          output(b, m, k) += lhs(b, m, n) * rhs(b, n, k);
        }
      }
    }
  }

  return output;
}

// ConcatenateOp
template <int64_t Dimension, typename Dest, typename Src>
inline Dest concatenate(Src input) {
  Dest z = input;
  return z;
}

template <int64_t Dimension, typename Dest, typename Src1, typename... Src>
inline Dest concatenate(Src1 input1, Src... inputs) {
  static_assert(sizeof...(inputs) > 0, "Wrong template specialization chosen");

  // Concatenate all but the first input.
  // We need to build the correct return type for the rest of the inputs.
  using ET_Src = typename get_element_type<Src1>::type;
  using Rest = typename concat<Dimension, ET_Src, Src...>::type;
  Rest rest = concatenate<Dimension, Rest, Src...>(inputs...);

  Dest z;

  // a: AxBxI    xD
  // b: AxBxJ    xD
  // c: AxBx(I+J)xD

  // repeat repeat until a_ptr == a.end():
  //    copy IxD elements from a_ptr to c_ptr
  //    move a_ptr, c_ptr by IxD elements
  //    copy JxD elements from b_ptr to c_ptr
  //    move b_ptr, c_ptr by JxD elements

  // Take the product of all dimensions, starting at `Dimension`.
  auto calculate_shift = [](const auto &shape) {
    size_t shift = 1;
    for (size_t i = Dimension; i < shape.size(); i++) {
      shift *= shape[i];
    }
    return shift;
  };
  auto a_shift = calculate_shift(Src1::shape());
  auto b_shift = calculate_shift(Rest::shape());

  for (auto a_ptr = input1.begin(), b_ptr = rest.begin(), c_ptr = z.begin();
       a_ptr != input1.end(); a_ptr += a_shift, b_ptr += b_shift) {
    std::copy(a_ptr, a_ptr + a_shift, c_ptr);
    c_ptr += a_shift;
    std::copy(b_ptr, b_ptr + b_shift, c_ptr);
    c_ptr += b_shift;
  }

  return z;
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

// SliceOp
// Overload for 1d case.
template <typename Dest, typename Src, IsTensorOfDim<1, Src> = true>
Dest slice(Src x, Tensor<int64_t, 1> start_indices,
           Tensor<int64_t, 1> limit_indices, Tensor<int64_t, 1> strides) {
  Dest z;

  size_t index = 0;
  for (int64_t i = start_indices[0]; i < limit_indices[0]; i += strides[0]) {
    z[index++] = x(i);
  }

  return z;
}

// Overload for 2d case.
template <typename Dest, typename Src, IsTensorOfDim<2, Src> = true>
Dest slice(Src x, Tensor<int64_t, 2> start_indices,
           Tensor<int64_t, 2> limit_indices, Tensor<int64_t, 2> strides) {
  Dest z;

  size_t index = 0;
  for (int64_t i = start_indices[0]; i < limit_indices[0]; i += strides[0]) {
    for (int64_t j = start_indices[1]; j < limit_indices[1]; j += strides[1]) {
      z[index++] = x(i, j);
    }
  }

  return z;
}

// Overload for 3d case.
template <typename Dest, typename Src, IsTensorOfDim<3, Src> = true>
Dest slice(Src x, Tensor<int64_t, 3> start_indices,
           Tensor<int64_t, 3> limit_indices, Tensor<int64_t, 3> strides) {
  Dest z;

  size_t index = 0;
  for (int64_t i = start_indices[0]; i < limit_indices[0]; i += strides[0]) {
    for (int64_t j = start_indices[1]; j < limit_indices[1]; j += strides[1]) {
      for (int64_t k = start_indices[2]; k < limit_indices[2];
           k += strides[2]) {
        z[index++] = x(i, j, k);
      }
    }
  }

  return z;
}

// Overload for 4d case.
template <typename Dest, typename Src, IsTensorOfDim<4, Src> = true>
Dest slice(Src x, Tensor<int64_t, 4> start_indices,
           Tensor<int64_t, 4> limit_indices, Tensor<int64_t, 4> strides) {
  Dest z;

  size_t index = 0;
  for (int64_t i = start_indices[0]; i < limit_indices[0]; i += strides[0]) {
    for (int64_t j = start_indices[1]; j < limit_indices[1]; j += strides[1]) {
      for (int64_t k = start_indices[2]; k < limit_indices[2];
           k += strides[2]) {
        for (int64_t c = start_indices[3]; c < limit_indices[3];
             c += strides[3]) {
          z[index++] = x(i, j, k, c);
        }
      }
    }
  }

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

#endif // EMITC_CORE_OPS_H
