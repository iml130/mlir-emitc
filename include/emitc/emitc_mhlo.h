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

// This file defines functions used by EmitC

#ifndef EMITC_MHLO_H
#define EMITC_MHLO_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <random>
#include <type_traits>

#include "emitc_tensor.h"

namespace mhlo {
/// See
/// https://github.com/tensorflow/tensorflow/blob/6f59650012f8904745dffaba540afc794c6613be/tensorflow/compiler/xla/service/hlo_evaluator.cc
/// for the XLA implementation

/// Functions for MHLO unary elementwise ops
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
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::log);

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

/// Functions for MHLO binary elementwise ops.
// AddOp
template <typename Src>
inline Src add(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::plus<ET_Src>{};

  return binary<Src>(x, y, f);
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
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::minus<ET_Src>{};

  return binary<Src>(x, y, f);
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
broadcast_in_dim(Src x, Tensor1D<int64_t, Src::rank()> broadcast_dimensions) {
  static_assert(Src::rank() == 0, "Only 0-dim operand supported so far");

  Dest z;
  std::fill(z.begin(), z.end(), x[0]);
  return z;
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
Dest slice(Src x, Tensor1D<int64_t, 1> start_indices,
           Tensor1D<int64_t, 1> limit_indices, Tensor1D<int64_t, 1> strides) {
  Dest z;

  size_t index = 0;
  for (int64_t i = start_indices[0]; i < limit_indices[0]; i += strides[0]) {
    z[index++] = x(i);
  }

  return z;
}

// Overload for 2d case
template <typename Dest, typename Src, IsTensorOfDim<2, Src> = true>
Dest slice(Src x, Tensor1D<int64_t, 2> start_indices,
           Tensor1D<int64_t, 2> limit_indices, Tensor1D<int64_t, 2> strides) {
  Dest z;

  size_t index = 0;
  for (int64_t i = start_indices[0]; i < limit_indices[0]; i += strides[0]) {
    for (int64_t j = start_indices[1]; j < limit_indices[1]; j += strides[1]) {
      z[index++] = x(i, j);
    }
  }

  return z;
}

// DynamicSliceOp
// Overload for 1d case
template <typename Dest, typename Src, IsTensorOfDim<1, Src> = true>
Dest dynamic_slice(Src x, int64_t start_index,
                   Tensor1D<int64_t, 1> size_indices) {
  auto clamp = [](int64_t value, int64_t minValue, int64_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  int64_t dim_x = static_cast<int64_t>(Src::dim(0));
  int64_t start_index_eff = clamp(start_index, 0, dim_x - size_indices[0]);
  Tensor1D<int64_t, 1> start_indices{start_index_eff};
  Tensor1D<int64_t, 1> limit_indices{start_index_eff + size_indices[0]};
  Tensor1D<int64_t, 1> strides{1};

  return slice<Dest, Src>(x, start_indices, limit_indices, strides);
}

// Overload for 2d case
template <typename Dest, typename Src, IsTensorOfDim<2, Src> = true>
Dest dynamic_slice(Src x, int64_t start_index_x, int64_t start_index_y,
                   Tensor1D<int64_t, 2> size_indices) {
  auto clamp = [](int64_t value, int64_t minValue, int64_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  int64_t dim_x = static_cast<int64_t>(Src::dim(0));
  int64_t dim_y = static_cast<int64_t>(Src::dim(1));
  int64_t start_index_x_eff = clamp(start_index_x, 0, dim_x - size_indices[0]);
  int64_t start_index_y_eff = clamp(start_index_y, 0, dim_y - size_indices[1]);
  Tensor1D<int64_t, 2> start_indices{start_index_x_eff, start_index_y_eff};
  Tensor1D<int64_t, 2> limit_indices{start_index_x_eff + size_indices[0],
                                     start_index_y_eff + size_indices[1]};
  Tensor1D<int64_t, 2> strides{1, 1};

  return slice<Dest, Src>(x, start_indices, limit_indices, strides);
}

// DynamicUpdateSliceOp
// Overload for 1d case
template <typename Update, typename Src, IsTensorOfDim<1, Src> = true>
Src dynamic_update_slice(Src x, Update update, int64_t start_index) {
  auto clamp = [](int64_t value, int64_t minValue, int64_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  Src z = x;

  size_t start_index_eff = clamp(start_index, 0, Src::dim(0) - Update::dim(0));

  for (size_t i = 0; i < Update::dim(0); i++) {
    z(start_index_eff + i) = update(i);
  }

  return z;
}

// Overload for 2d case
template <typename Update, typename Src, IsTensorOfDim<2, Src> = true>
Src dynamic_update_slice(Src x, Update update, int64_t start_index_x,
                         int64_t start_index_y) {
  auto clamp = [](int64_t value, int64_t minValue, int64_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  Src z = x;

  size_t start_index_x_eff =
      clamp(start_index_x, 0, Src::dim(0) - Update::dim(0));
  size_t start_index_y_eff =
      clamp(start_index_y, 0, Src::dim(1) - Update::dim(1));

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
inline Dest rng_uniform(T low, T high, Tensor1D<int64_t, N> shape) {
  static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                "Expected integer or floating point type");
  using uniform_distribution =
      typename std::conditional<std::is_integral<T>::value,
                                std::uniform_int_distribution<T>,
                                std::uniform_real_distribution<T>>::type;

  // high value is exclusive in xla but inclusive in cpp
  // see https://www.tensorflow.org/xla/operation_semantics?hl=en#rnguniform and
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
  if (std::is_integral<T>::value) {
    high = high - 1;
  }

  uniform_distribution distribution(low, high);
  std::random_device rd;
  std::mt19937 gen(rd());

  Dest z;

  for (size_t i = 0; i < z.size(); i++) {
    z[i] = distribution(gen);
  }

  return z;
}

// RngBitGeneratorOp
template <typename Dest>
Dest rng_bit_generator(std::tuple_element<0, Dest> state) {
  // TODO implement correct algorithm; starting point would be
  // https://github.com/tensorflow/tensorflow/blob/6f59650012f8904745dffaba540afc794c6613be/tensorflow/compiler/xla/service/rng_bit_generator_expander.cc#L56

  using StateType = std::tuple_element<0, Dest>;
  using TensorType = std::tuple_element<1, Dest>;
  using T = typename TensorType::value_type;

  StateType newState(state);

  T min = std::numeric_limits<T>::min();
  T max = std::numeric_limits<T>::max();
  TensorType data = rng_uniform<TensorType, T>(min, max, TensorType::shape());

  return std::make_tuple(newState, data);
}

} // namespace mhlo

#endif // EMITC_MHLO_H
