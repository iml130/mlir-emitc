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
//
// SPDX-License-Identifier: Apache-2.0

// This file defines functions emitted by MHLOToEmitC.

#ifndef EMITC_MHLO_H
#define EMITC_MHLO_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

#include "emitc/core_ops.h"

namespace emitc {
namespace mhlo {
/// See
/// https://github.com/tensorflow/tensorflow/blob/6f59650012f8904745dffaba540afc794c6613be/tensorflow/compiler/xla/service/hlo_evaluator.cc
/// for the XLA implementation

/// Functions for MHLO unary elementwise ops.
// AbsOp
// TODO: Add support for complex numbers.
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
  return emitc::convert<Dest>(x);
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
  return emitc::negate(x);
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
  return emitc::max(x, y);
}

// MinOp
template <typename Src>
inline Src min(Src x, Src y) {
  return emitc::min(x, y);
}

// MulOp
template <typename Src>
inline Src mul(Src x, Src y) {
  return emitc::mul(x, y);
}

// PowOp
template <typename Src>
inline Src pow(Src x, Src y) {
  return emitc::pow(x, y);
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
// The broadcast_dimensions argument maps from Src to Dest dimensions.
template <typename Dest, typename Src>
inline Dest
broadcast_in_dim(Src operand,
                 Tensor<int64_t, Src::rank()> broadcast_dimensions) {
  return emitc::broadcast_in_dim<Dest>(operand, broadcast_dimensions);
}

// ClampOp
template <typename Min, typename Src, typename Max>
inline Src clamp(Min min, Src operand, Max max) {
  return emitc::clamp(min, operand, max);
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

// SliceOp
template <typename Dest, typename Src>
Dest slice(Src x, Tensor<int64_t, Src::rank()> start_indices,
           Tensor<int64_t, Src::rank()> limit_indices,
           Tensor<int64_t, Src::rank()> strides) {
  return emitc::slice<Dest, Src>(x, start_indices, limit_indices, strides);
}

// DynamicSliceOp
// Overload for 1d case.
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

// Overload for 2d case.
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
// Overload for 1d case.
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

// Overload for 2d case.
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
  return emitc::reshape<Dest>(x);
}

// PadOp
// TODO: Support negative edge padding.
template <typename Dest, typename Src>
inline Dest pad(Src operand,
                Tensor<typename get_element_type<Src>::type> padding_value,
                Tensor<int64_t, Src::rank()> edge_padding_low,
                Tensor<int64_t, Src::rank()> edge_padding_high,
                Tensor<int64_t, Src::rank()> interior_padding) {
  return emitc::pad<Dest>(operand, padding_value, edge_padding_low,
                          edge_padding_low, interior_padding);
}

// ReduceOp
// 1 result overload
template <typename Dest, size_t Dimension, typename Src, typename Computation>
inline Dest
reduce(Src operand, Tensor<typename get_element_type<Src>::type> initValue,
       Tensor<int64_t, Dimension> dimensions, Computation computation) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Dest>::type;

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

// 2 result overload
template <typename Dest1, typename Dest2, size_t Dimension, typename Src1,
          typename Src2, typename Computation>
inline std::tuple<Dest1, Dest2>
reduce(Src1 operand1, Src2 operand2,
       Tensor<typename get_element_type<Src1>::type> initValue1,
       Tensor<typename get_element_type<Src2>::type> initValue2,
       Tensor<int64_t, Dimension> dimensions, Computation computation) {
  static_assert(is_tensor<Src1>::value, "Expected tensor argument");
  static_assert(is_tensor<Src2>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest1>::value, "Expected tensor result");
  static_assert(is_tensor<Dest2>::value, "Expected tensor result");

  using ET_Src1 = typename get_element_type<Src1>::type;
  using ET_Src2 = typename get_element_type<Src2>::type;
  using ET_Dest1 = typename get_element_type<Dest1>::type;
  using ET_Dest2 = typename get_element_type<Dest2>::type;

  static_assert(std::is_same<ET_Src1, ET_Dest2>::value,
                "Element type mismatch");
  static_assert(std::is_same<ET_Src2, ET_Dest2>::value,
                "Element type mismatch");

  static_assert(Src1::rank() == Dest1::rank() + Dimension,
                "source rank must equal dest rank + dimension size");
  static_assert(Src2::rank() == Dest2::rank() + Dimension,
                "source rank must equal dest rank + dimension size");

  static_assert(Src1::rank() == Src2::rank(), "source ranks must match");
  static_assert(Dest1::rank() == Dest2::rank(), "destination ranks must match");

  std::vector<size_t> retainedDimensions(Src1::rank());
  std::iota(retainedDimensions.begin(), retainedDimensions.end(), 0);

  retainedDimensions.erase(
      std::remove_if(retainedDimensions.begin(), retainedDimensions.end(),
                     [&dimensions](size_t i) {
                       return std::find(dimensions.begin(), dimensions.end(),
                                        i) != dimensions.end();
                     }),
      retainedDimensions.end());

  assert(retainedDimensions.size() == Dest1::rank());

  Dest1 result1;
  Dest2 result2;
  std::fill(result1.begin(), result1.end(), initValue1());
  std::fill(result2.begin(), result2.end(), initValue2());

  for (size_t i = 0; i < operand1.size(); i++) {
    auto index = operand1.unravel_index(i);
    auto value1 = Tensor<ET_Src1>{operand1[i]};
    auto value2 = Tensor<ET_Src2>{operand2[i]};

    std::array<size_t, Dest1::rank()> reducedIndex;
    size_t j = 0;
    for (size_t dim : retainedDimensions) {
      reducedIndex[j++] = index[dim];
    }

    auto reductionValue1 =
        Tensor<ET_Src1>{result1[result1.ravel_index(reducedIndex)]};
    auto reductionValue2 =
        Tensor<ET_Src1>{result2[result2.ravel_index(reducedIndex)]};
    Tensor<ET_Dest1> resultValue1;
    Tensor<ET_Dest2> resultValue2;
    std::tie(resultValue1, resultValue2) =
        computation(reductionValue1, value1, reductionValue2, value2);

    result1[result1.ravel_index(reducedIndex)] = resultValue1();
    result2[result2.ravel_index(reducedIndex)] = resultValue2();
  }

  return std::make_tuple(result1, result2);
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

    // Iterate over input window.
    for (auto &inputIndex : operand.window(baseIndex, windowDimensionsArr)) {
      // Get input value (check out of bounds access).
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

      // Get reduction value.
      auto reductionValue = Tensor<ET_Src>{result[result.ravel_index(index)]};
      // Run computation.
      Tensor<ET_Dest> resultValue = computation(reductionValue, value);

      // Update result value.
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
inline Src select(Tensor<bool> pred, Src on_true, Src on_false) {
  Src z;

  for (size_t i = 0; i < Src::size(); i++) {
    z[i] = pred[0] ? on_true[i] : on_false[i];
  }

  return z;
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

  // High value is exclusive in XLA but inclusive in cpp
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
// TODO: Replicate ConvDimensionNumbers struct.
// TODO: Implement general dimension numbers.
// TODO: Implement lhs_dilation.
// TODO: Implement rhs_dilation.
// TODO: Implement batch_group_count.
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

  assert(window_strides[0] > 0);
  assert(window_strides[1] > 0);

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

  // TODO: Test grouped convolutions.
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
} // namespace emitc

#endif // EMITC_MHLO_H
