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

// This file defines functions emitted by MHLOToEmitC

#ifndef EMITC_EMITC_MHLO_H
#define EMITC_EMITC_MHLO_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

#include "emitc_core_ops.h"

namespace mhlo {
/// See
/// https://github.com/tensorflow/tensorflow/blob/6f59650012f8904745dffaba540afc794c6613be/tensorflow/compiler/xla/service/hlo_evaluator.cc
/// for the XLA implementation

/// Functions for MHLO unary elementwise ops
// AbsOp
// TODO support complex numbers
template <typename Src>
inline Src abs(Src x) {
  return emitc::abs<Src>(x);
}

// CeilOp
template <typename Src>
inline Src ceil(Src x) {
  return emitc::ceil<Src>(x);
}

// BitcastConvertOp
template <typename Dest, typename Src>
inline Dest bitcast_convert(Src x) {
  using ET_Dest = typename get_element_type<Dest>::type;
  using ET_Src = typename get_element_type<Src>::type;

  static_assert(sizeof(ET_Src) == sizeof(ET_Dest),
                "Can only bitcast on types of the same size");

  auto cast = [](ET_Src value) {
    ET_Dest result;
    memcpy(&result, &value, sizeof(ET_Src));
    return result;
  };

  return unary<Dest, Src, UnaryFuncType<ET_Dest, ET_Src>>(x, cast);
}

// CompareOp
template <typename Src, template <typename> class Compare>
typename replace_element_type<bool, Src>::type compare(Src x, Src y) {
  using Dest = typename replace_element_type<bool, Src>::type;
  using ET_Src = typename get_element_type<Src>::type;

  auto cmp = Compare<ET_Src>{};

  return binary<Dest, Src>(x, y, cmp);
}

// ConvertOp
template <typename Dest, typename Src>
inline Dest convert(Src x) {
  using ET_Dest = typename get_element_type<Dest>::type;
  using ET_Src = typename get_element_type<Src>::type;

  auto cast = [](ET_Src value) { return static_cast<ET_Dest>(value); };

  return unary<Dest, Src, UnaryFuncType<ET_Dest, ET_Src>>(x, cast);
}

// CosOp
template <typename Src>
inline Src cos(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::cos);

  return unary<Src>(x, f);
}

// ExpOp
template <typename Src>
inline Src exponential(Src x) {
  return emitc::exp<Src>(x);
}

// Expm1Op
template <typename Src>
inline Src exponential_minus_one(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::expm1);

  return unary<Src>(x, f);
}

// FloorOp
template <typename Src>
inline Src floor(Src x) {
  return emitc::floor<Src>(x);
}

// IsFiniteOp
template <typename Src>
inline typename replace_element_type<bool, Src>::type is_finite(Src x) {
  using ET_Src = typename get_element_type<Src>::type;
  static_assert(std::is_floating_point<ET_Src>::value,
                "Operation supports only floating point types");

  using Dest = typename replace_element_type<bool, Src>::type;

  auto f = static_cast<bool (*)(ET_Src)>(std::isfinite);

  return unary<Dest, Src>(x, f);
}

// LogOp
template <typename Src>
inline Src log(Src x) {
  return emitc::log<Src>(x);
}

// Log1pOp
template <typename Src>
inline Src log_plus_one(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::log1p);

  return unary<Src>(x, f);
}

// NegOp
template <typename Src>
inline Src negate(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::negate<ET_Src>{};

  return unary<Src>(x, f);
}

// RoundOp
template <typename Src>
inline Src round(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::round);

  return unary<Src>(x, f);
}

// SinOp
template <typename Src>
inline Src sin(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::sin);

  return unary<Src>(x, f);
}

// SqrtOp
template <typename Src>
inline Src sqrt(Src x) {
  return emitc::sqrt<Src>(x);
}

// TanhOp
template <typename Src>
inline Src tanh(Src x) {
  return emitc::tanh<Src>(x);
}

/// Functions for MHLO binary elementwise ops.
// AddOp
template <typename Src>
inline Src add(Src x, Src y) {
  return emitc::add<Src>(x, y);
}

// Atan2Op
template <typename Src>
inline Src atan2(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src, ET_Src)>(std::atan2);

  return binary<Src>(x, y, f);
}

// DivOp
template <typename Src>
inline Src div(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::divides<ET_Src>{};

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
  return emitc::mul(x, y);
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

// ShiftLeftOp
template <typename Src>
inline Src shift_left(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;
  static_assert(std::is_unsigned<ET_Src>::value,
                "Operation not implemented for signed types");

  auto f = [](ET_Src a, ET_Src b) -> ET_Src { return a << b; };

  return binary<Src>(x, y, f);
}

// ShiftRightLogicalOp
template <typename Src>
inline Src shift_right_logical(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;
  static_assert(std::is_unsigned<ET_Src>::value,
                "Operation not implemented for signed types");

  auto f = [](ET_Src a, ET_Src b) -> ET_Src { return a >> b; };

  return binary<Src>(x, y, f);
}

// SubOp
template <typename Src>
inline Src sub(Src x, Src y) {
  return emitc::sub<Src>(x, y);
}

/// Functions for MHLO binary logical elementwise ops.
// OrOp
template <typename Src>
inline Src logical_or(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::logical_or<ET_Src>{};

  return binary<Src>(x, y, f);
}

// XorOp
template <typename Src>
inline Src logical_xor(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = [](ET_Src a, ET_Src b) -> ET_Src { return a != b; };

  return binary<Src>(x, y, f);
}

/// Functions for other MHLO ops.
// BroadcastInDimOp
template <typename Dest, typename Src>
inline Dest
broadcast_in_dim(Src operand,
                 Tensor<int64_t, Src::rank()> broadcast_dimensions) {
  return emitc::broadcast_in_dim<Dest>(operand, broadcast_dimensions);
}

// ClampOp
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
                        typename get_element_type<Min>::type>::value),
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

// ConcatenateOp
template <int64_t Dimension, typename Dest, typename Src>
inline Dest concatenate(Src input) {
  Dest z = input;
  return z;
}

template <int64_t Dimension, typename Dest, typename Src1, typename... Src>
inline Dest concatenate(Src1 input1, Src... inputs) {
  static_assert(sizeof...(inputs) > 0, "Wrong template specialization chosen");

  // concatenate all but the first input
  // We need to build the correct return type for the rest of the inputs
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

  // take the product of all dimensions, starting at `Dimension`
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

// SliceOp
// Overload for 1d case
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

// Overload for 2d case
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

// Overload for 3d case
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

// Overload for 4d case
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

// DynamicSliceOp
// Overload for 1d case
template <typename Dest, typename Src, IsTensorOfDim<1, Src> = true>
Dest dynamic_slice(Src x, Tensor<int32_t> start_index,
                   Tensor<int64_t, 1> slice_sizes) {
  auto clamp = [](int64_t value, int64_t minValue, int64_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  int64_t dim_x = static_cast<int64_t>(Src::dim(0));
  int64_t start_index_eff = clamp(start_index[0], 0, dim_x - slice_sizes[0]);
  Tensor<int64_t, 1> start_indices{start_index_eff};
  Tensor<int64_t, 1> limit_indices{start_index_eff + slice_sizes[0]};
  Tensor<int64_t, 1> strides{1};

  return slice<Dest, Src>(x, start_indices, limit_indices, strides);
}

// Overload for 2d case
template <typename Dest, typename Src, IsTensorOfDim<2, Src> = true>
Dest dynamic_slice(Src x, Tensor<int32_t> start_index_x,
                   Tensor<int32_t> start_index_y,
                   Tensor<int64_t, 2> slice_sizes) {
  auto clamp = [](int64_t value, int64_t minValue, int64_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  int64_t dim_x = static_cast<int64_t>(Src::dim(0));
  int64_t dim_y = static_cast<int64_t>(Src::dim(1));
  int64_t start_index_x_eff =
      clamp(start_index_x[0], 0, dim_x - slice_sizes[0]);
  int64_t start_index_y_eff =
      clamp(start_index_y[0], 0, dim_y - slice_sizes[1]);
  Tensor<int64_t, 2> start_indices{start_index_x_eff, start_index_y_eff};
  Tensor<int64_t, 2> limit_indices{start_index_x_eff + slice_sizes[0],
                                   start_index_y_eff + slice_sizes[1]};
  Tensor<int64_t, 2> strides{1, 1};

  return slice<Dest, Src>(x, start_indices, limit_indices, strides);
}

// DynamicUpdateSliceOp
// Overload for 1d case
template <typename Update, typename Src, IsTensorOfDim<1, Src> = true>
Src dynamic_update_slice(Src x, Update update, Tensor<int32_t> start_index) {
  auto clamp = [](int64_t value, int64_t minValue, int64_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  Src z = x;

  size_t start_index_eff =
      clamp(start_index[0], 0, Src::dim(0) - Update::dim(0));

  for (size_t i = 0; i < Update::dim(0); i++) {
    z(start_index_eff + i) = update(i);
  }

  return z;
}

// Overload for 2d case
template <typename Update, typename Src, IsTensorOfDim<2, Src> = true>
Src dynamic_update_slice(Src x, Update update, Tensor<int32_t> start_index_x,
                         Tensor<int32_t> start_index_y) {
  auto clamp = [](int64_t value, int64_t minValue, int64_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  Src z = x;

  size_t start_index_x_eff =
      clamp(start_index_x[0], 0, Src::dim(0) - Update::dim(0));
  size_t start_index_y_eff =
      clamp(start_index_y[0], 0, Src::dim(1) - Update::dim(1));

  for (size_t i = 0; i < Update::dim(0); i++) {
    for (size_t j = 0; j < Update::dim(1); j++) {
      z(start_index_x_eff + i, start_index_y_eff + j) = update(i, j);
    }
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

// PadOp
// TODO support negative edge padding
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

    // shift by low padding
    for (size_t j = 0; j < index.size(); j++) {
      index[j] -= edge_padding_low[j];
    }

    if (interior(index)) {
      result[i] = padding_value();
    } else {
      // squeeze by interrior padding
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

// ReduceOp
template <typename Dest, size_t Dimension, typename Src, typename Computation>
inline Dest
reduce(Src operand, Tensor<typename get_element_type<Src>::type> initValue,
       Tensor<int64_t, Dimension> dimensions, Computation computation) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Src>::type;

  static_assert(std::is_same<ET_Src, ET_Dest>::value, "Element type mismatch");

  static_assert(Src::rank() == Dest::rank() + Dimension,
                "source rank must equal dest rank + dimension size");

  std::vector<size_t> retainedDimensions(Src::rank());
  std::iota(retainedDimensions.begin(), retainedDimensions.end(), 0);

  retainedDimensions.erase(
      std::remove_if(retainedDimensions.begin(), retainedDimensions.end(),
                     [&dimensions](size_t i) {
                       return std::find(dimensions.begin(), dimensions.end(),
                                        i) != dimensions.end();
                     }),
      retainedDimensions.end());

  assert(retainedDimensions.size() == Dest::rank());

  Dest result;
  std::fill(result.begin(), result.end(), initValue());

  for (size_t i = 0; i < operand.size(); i++) {
    auto value = Tensor<ET_Src>{operand[i]};
    auto index = operand.unravel_index(i);

    std::array<size_t, Dest::rank()> reducedIndex;
    size_t j = 0;
    for (size_t dim : retainedDimensions) {
      reducedIndex[j++] = index[dim];
    }

    auto reductionValue =
        Tensor<ET_Src>{result[result.ravel_index(reducedIndex)]};
    Tensor<ET_Dest> resultValue = computation(reductionValue, value);

    result[result.ravel_index(reducedIndex)] = resultValue();
  }

  return result;
}

// ReduceWindowOp
template <typename Dest, typename Src, typename Computation>
inline Dest reduce_window(
    Src operand, Tensor<typename get_element_type<Src>::type> initValue,
    Tensor<int64_t, Src::rank()> window_dimensions,
    Tensor<int64_t, Src::rank()> window_strides,
    Tensor<int64_t, Src::rank()> base_dilations,
    Tensor<int64_t, Src::rank()> window_dilations,
    Tensor<int64_t, 2, Src::rank()> padding, Computation computation) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Src>::type;

  static_assert(std::is_same<ET_Src, ET_Dest>::value, "Element type mismatch");
  static_assert(Src::rank() == Dest::rank(), "Rank mismatch");

  assert(std::all_of(window_dimensions.begin(), window_dimensions.end(),
                     [](int64_t i) { return i > 0; }));
  assert(std::all_of(base_dilations.begin(), base_dilations.end(),
                     [](int64_t i) { return i == 1; }));
  assert(std::all_of(window_dilations.begin(), window_dilations.end(),
                     [](int64_t i) { return i == 1; }));

  auto out_of_bounds = [&padding](std::array<size_t, Src::rank()> index) {
    for (size_t i = 0; i < index.size(); i++) {
      if (index[i] < static_cast<size_t>(padding(0, i)) ||
          index[i] >= Src::dim(i) + static_cast<size_t>(padding(0, i))) {
        return true;
      }
    }
    return false;
  };

  std::array<size_t, Src::rank()> windowDimensionsArr;
  for (size_t j = 0; j < windowDimensionsArr.size(); j++) {
    windowDimensionsArr[j] = static_cast<size_t>(window_dimensions[j]);
  }

  Dest result;
  std::fill(result.begin(), result.end(), initValue());

  for (size_t i = 0; i < result.size(); i++) {
    auto index = result.unravel_index(i);

    std::array<size_t, Src::rank()> baseIndex;
    for (size_t j = 0; j < baseIndex.size(); j++) {
      baseIndex[j] = index[j] * window_strides(j);
    }

    // iterate over input window
    for (auto &inputIndex : operand.window(baseIndex, windowDimensionsArr)) {
      // get input value (check out of bounds access)
      Tensor<ET_Src> value;
      if (out_of_bounds(inputIndex)) {
        value[0] = initValue[0];
      } else {
        std::array<size_t, Src::rank()> _index;
        for (size_t j = 0; j < inputIndex.size(); j++) {
          assert(inputIndex[j] >= static_cast<size_t>(padding(0, j)));
          _index[j] = inputIndex[j] - static_cast<size_t>(padding(0, j));
        }
        value[0] = operand[operand.ravel_index(_index)];
      }

      // get reduction value
      auto reductionValue = Tensor<ET_Src>{result[result.ravel_index(index)]};
      // run computation
      Tensor<ET_Dest> resultValue = computation(reductionValue, value);

      // update result value
      result[result.ravel_index(index)] = resultValue();
    }
  }

  return result;
}

// SelectOp
template <typename Src, IsScalar<Src> = true>
inline Src select(typename replace_element_type<bool, Src>::type pred,
                  Src on_true, Src on_false) {
  return pred ? on_true : on_false;
}

template <typename Src, IsTensor<Src> = true>
inline Src select(typename replace_element_type<bool, Src>::type pred,
                  Src on_true, Src on_false) {
  Src z;

  for (size_t i = 0; i < Src::size(); i++) {
    z[i] = pred[i] ? on_true[i] : on_false[i];
  }

  return z;
}

// RngUniformOp
template <typename Dest, typename T, size_t N>
inline Dest rng_uniform(Tensor<T> low, Tensor<T> high,
                        Tensor<int64_t, N> shape) {
  static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                "Expected integer or floating point type");
  using uniform_distribution =
      typename std::conditional<std::is_integral<T>::value,
                                std::uniform_int_distribution<T>,
                                std::uniform_real_distribution<T>>::type;
  T lowValue = low[0];
  T highValue = high[0];

  // high value is exclusive in xla but inclusive in cpp
  // see https://www.tensorflow.org/xla/operation_semantics?hl=en#rnguniform
  // and
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
  if (std::is_integral<T>::value) {
    highValue = highValue - 1;
  }

  uniform_distribution distribution(lowValue, highValue);
  std::random_device rd;
  std::mt19937 gen(rd());

  Dest z;

  for (size_t i = 0; i < z.size(); i++) {
    z[i] = distribution(gen);
  }

  return z;
}

// RngBitGeneratorOp
template <typename Dest, int32_t RngAlgorithm>
Dest rng_bit_generator(typename std::tuple_element<0, Dest>::type state) {
  // TODO implement correct algorithm; starting point would be
  // https://github.com/tensorflow/tensorflow/blob/6f59650012f8904745dffaba540afc794c6613be/tensorflow/compiler/xla/service/rng_bit_generator_expander.cc#L56

  using StateType = typename std::tuple_element<0, Dest>::type;
  using TensorType = typename std::tuple_element<1, Dest>::type;
  using T = typename TensorType::value_type;

  StateType newState(state);

  T minValue = std::numeric_limits<T>::min();
  T maxValue = std::numeric_limits<T>::max();

  Tensor<T> min{minValue};
  Tensor<T> max{maxValue};

  std::array<size_t, TensorType::rank()> arrayShape = TensorType::shape();
  Tensor<int64_t, TensorType::rank()> tensorShape;

  for (size_t i = 0; i < TensorType::rank(); i++) {
    tensorShape[i] = static_cast<int64_t>(arrayShape[i]);
  }

  TensorType data = rng_uniform<TensorType, T>(min, max, tensorShape);

  return std::make_tuple(newState, data);
}

// BatchNormInferenceOp
template <typename Src, typename Feature>
Src batch_norm_inference(Src input, Feature scale, Feature offset, Feature mean,
                         Feature variance, float epsilon,
                         int64_t feature_index) {
  static_assert(is_tensor_of_dim<1, Feature>::value,
                "Expected 1 dimensional statistic features");
  assert(0 <= feature_index &&
         static_cast<size_t>(feature_index) < Src::rank());
  assert(Src::dim(feature_index) == Feature::dim(0));
  assert(epsilon > 0);

  Src output;
  for (size_t i = 0; i < Src::size(); i++) {
    auto multi_index = input.unravel_index(i);
    size_t f_index = multi_index[feature_index];
    auto value = (input[i] - mean[f_index]) / sqrt(variance[f_index] + epsilon);
    output[i] = value * scale[f_index] + offset[f_index];
  }
  return output;
}

// ConvolutionOp
// TODO replicate ConvDimensionNumbers struct
// TODO implement general dimension numbers
// TODO implement lhs_dilation
// TODO implement rhs_dilation
// TODO implement batch_group_count
template <typename Dest, typename Src, typename Weights>
Dest convolution(Src input, Weights weights, int64_t batch_group_count,
                 int64_t input_batch_dimension, int64_t input_feature_dimension,
                 Tensor<int64_t, 2> input_spatial_dimensions,
                 int64_t kernel_input_feature_dimension,
                 int64_t kernel_output_feature_dimension,
                 Tensor<int64_t, 2> kernel_spatial_dimensions,
                 int64_t output_batch_dimension,
                 int64_t output_feature_dimension,
                 Tensor<int64_t, 2> output_spatial_dimensions,
                 int64_t feature_group_count, Tensor<int64_t, 2, 2> padding,
                 Tensor<int64_t, 2> lhs_dilation,
                 Tensor<int64_t, 2> rhs_dilation,
                 Tensor<int64_t, 2> window_strides) {
  static_assert(is_tensor_of_dim<4, Src>::value,
                "Expected 4 dimensional input");
  static_assert(is_tensor_of_dim<4, Dest>::value,
                "Expected 4 dimensional output");
  static_assert(is_tensor_of_dim<4, Weights>::value,
                "Expected 4 dimensional weights");

  assert(batch_group_count == 1);

  assert(input_batch_dimension == 0);
  assert(input_spatial_dimensions[0] == 1);
  assert(input_spatial_dimensions[1] == 2);
  assert(input_feature_dimension == 3);

  assert(kernel_spatial_dimensions[0] == 0);
  assert(kernel_spatial_dimensions[1] == 1);
  assert(kernel_input_feature_dimension == 2);
  assert(kernel_output_feature_dimension == 3);

  assert(output_batch_dimension == 0);
  assert(output_spatial_dimensions[0] == 1);
  assert(output_spatial_dimensions[1] == 2);
  assert(output_feature_dimension == 3);

  assert(input.dim(input_feature_dimension) % feature_group_count == 0);
  assert(weights.dim(kernel_input_feature_dimension) ==
         input.dim(input_feature_dimension) / feature_group_count);

  assert(lhs_dilation[0] == 1);
  assert(lhs_dilation[0] == 1);

  assert(rhs_dilation[0] == 1);
  assert(rhs_dilation[0] == 1);

  const int N = input.dim(input_batch_dimension);
  const int H_IN = input.dim(input_spatial_dimensions[0]);
  const int W_IN = input.dim(input_spatial_dimensions[1]);
  const int C_IN = input.dim(input_feature_dimension);

  assert(C_IN % feature_group_count == 0);
  const int G_IN = C_IN / feature_group_count;

  Dest output;

  const int C_OUT = output.dim(output_feature_dimension);
  assert(C_OUT % feature_group_count == 0);
  const int G_OUT = C_OUT / feature_group_count;

  const int K_H = weights.dim(kernel_spatial_dimensions[0]);
  const int K_W = weights.dim(kernel_spatial_dimensions[1]);
  const int S_H = window_strides[0];
  const int S_W = window_strides[1];

  const int pt = padding(0, 0);
  const int pb = padding(0, 1);
  const int pl = padding(1, 0);
  const int pr = padding(1, 1);

  const int H_PAD = pt + H_IN + pb;
  const int W_PAD = pl + W_IN + pr;

  // TODO test grouped convolutions
  assert(feature_group_count == 1 || feature_group_count == C_OUT);

  // Convolution
  for (int n = 0; n < N; n++) {
    for (int h_pad = 0; h_pad < H_PAD - K_H + 1; h_pad += S_H) {
      for (int w_pad = 0; w_pad < W_PAD - K_W + 1; w_pad += S_W) {
        for (int kh = 0; kh < K_H; kh++) {
          for (int kw = 0; kw < K_W; kw++) {
            for (int g = 0; g < feature_group_count; g++) {
              for (int g_in = 0; g_in < G_IN; g_in++) {
                for (int g_out = 0; g_out < G_OUT; g_out++) {
                  const int h_out = h_pad / S_H;
                  const int w_out = w_pad / S_W;
                  const int c_out = g * G_OUT + g_out;
                  const int h_in = h_pad - pt + kh;
                  const int w_in = w_pad - pl + kw;
                  const int c_in = g * G_IN + g_in;

                  if (h_in < 0 || h_in >= H_IN || w_in < 0 || w_in >= W_IN)
                    continue;
                  output(n, h_out, w_out, c_out) +=
                      input(n, h_in, w_in, c_in) * weights(kh, kw, g_in, c_out);
                }
              }
            }
          }
        }
      }
    }
  }
  return output;
}

// DotOp
template <typename Dest, typename Lhs, typename Rhs>
Dest dot(Lhs lhs, Rhs rhs) {
  return emitc::dot<Dest>(lhs, rhs);
}

} // namespace mhlo

#endif // EMITC_EMITC_MHLO_H
