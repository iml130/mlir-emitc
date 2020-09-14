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
#include <vector>

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

// BitcastConvertOp
template <typename T1, typename T2>
inline T1 bitcast_convert(T2 x) {
  return reinterpret_cast<T1>(x);
}

template <typename T1, typename T2>
inline std::vector<T1> bitcast_convert(std::vector<T2> x) {
  std::vector<T1> z(x.size());
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = reinterpret_cast<T1>(x[i]);
  }
  return z;
}

// CompareOp
template <typename T, template <typename> class Compare>
std::vector<bool> compare(std::vector<T> x, std::vector<T> y) {
  std::vector<bool> z(x.size());
  std::transform(x.begin(), x.end(), y.begin(), z.begin(), Compare<T>());
  return z;
}

// ConvertOp
template <typename T1, typename T2>
inline T1 convert(T2 x) {
  return static_cast<T1>(x);
}

template <typename T1, typename T2>
inline std::vector<T1> convert(std::vector<T2> x) {
  std::vector<T1> z(x.size());
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = static_cast<T1>(x[i]);
  }
  return z;
}

// CosOp
template <typename Src>
inline Src cos(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::cos);

  return unary<Src>(x, f);
}

// ExpOp
// TODO: Implement!
// `e^(operand)` element-wise

// IsFiniteOp
// TODO: Implement!

// LogOp
// TODO: Implement!

// NegOp
template <typename Src>
inline Src negate(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::negate<ET_Src>{};

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
template <typename T>
inline T sqrt(T x) {
  return std::sqrt(x);
}

template <typename T>
inline std::vector<T> sqrt(std::vector<T> x) {
  std::vector<T> z(x);
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = std::sqrt(x[i]);
  }
  return z;
}

/// Functions for MHLO binary elementwise ops.
// AddOp
template <typename Src>
inline Src add(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::plus<ET_Src>{};

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
template <typename T1, typename T2>
inline T1 pow(T1 x, T2 y) {
  return std::pow(x, y);
}

template <typename T>
inline std::vector<T> pow(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  std::transform(x.begin(), x.end(), y.begin(), z.begin(),
                 [](auto a, auto b) { return std::pow(a, b); });
  return z;
}

// ShiftLeftOp
// TODO: Implement!

// ShiftRightLogicalOp
// TODO: Implement!

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
template <typename T>
inline std::vector<T> broadcast_in_dim(std::vector<T> x, size_t n) {
  std::vector<T> z;

  for (size_t i = 0; i < n; i++) {
    z.insert(z.end(), x.begin(), x.end());
  }

  return z;
}

// ConcatenateOp
template <typename T>
inline std::vector<T> concatenate(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  z.insert(z.end(), y.begin(), y.end());
  return z;
}

// SliceOp
// Overload for 1d case
template <typename T, size_t Start, size_t Limit, size_t Stride,
          size_t InputShape, size_t OutputShape>
std::vector<T> slice(std::vector<T> x) {
  std::vector<T> result(OutputShape);

  size_t idx = 0;
  for (size_t i = Start; i < Limit; i += Stride) {
    result[idx++] = x[i];
  }
  return result;
}

// Overload for 2d case
template <typename T, size_t Start1, size_t Start2, size_t Limit1,
          size_t Limit2, size_t Stride1, size_t Stride2, size_t InputShape1,
          size_t InputShape2, size_t OutputShape1, size_t OutputShape2>
std::vector<T> slice(std::vector<T> x) {
  std::vector<T> result(OutputShape1 * OutputShape2);

  size_t idx = 0;
  for (size_t i = Start1; i < Limit1; i += Stride1) {
    for (size_t j = Start2; j < Limit2; j += Stride2) {
      result[idx++] = x[i * InputShape2 + j];
    }
  }
  return result;
}

// DynamicSliceOp
// Overload for 1d case
template <typename T, size_t Size, size_t InputShape, size_t OutputShape>
std::vector<T> dynamic_slice(std::vector<T> x,
                             std::vector<int64_t> startIndex) {
  std::vector<T> result(OutputShape);

  auto clamp = [](size_t value, size_t minValue, size_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  size_t startIndex_ = startIndex[0];
  startIndex_ = clamp(startIndex_, 0, InputShape - Size);

  size_t limit = startIndex_ + Size;

  size_t idx = 0;
  for (size_t i = startIndex_; i < limit; i++) {
    result[idx++] = x[i];
  }
  return result;
}

// Overload for 2d case
template <typename T, size_t SizeX, size_t SizeY, size_t InputShapeX,
          size_t InputShapeY, size_t OutputShapeX, size_t OutputShapeY>
std::vector<T> dynamic_slice(std::vector<T> x, std::vector<int64_t> startIndexX,
                             std::vector<int64_t> startIndexY) {
  std::vector<T> result(OutputShapeX * OutputShapeY);

  auto clamp = [](size_t value, size_t minValue, size_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  size_t startIndexX_ = startIndexX[0];
  size_t startIndexY_ = startIndexY[0];
  startIndexX_ = clamp(startIndexX_, 0, InputShapeX - SizeX);
  startIndexY_ = clamp(startIndexY_, 0, InputShapeY - SizeY);

  size_t limitX = startIndexX_ + SizeX;
  size_t limitY = startIndexY_ + SizeY;

  size_t idx = 0;
  for (size_t i = startIndexX_; i < limitX; i++) {
    for (size_t j = startIndexY_; j < limitY; j++) {
      result[idx++] = x[i * InputShapeY + j];
    }
  }
  return result;
}

// DynamicUpdateSliceOp
// Overload for 1d case
template <typename T, size_t InputShape, size_t UpdateShape>
std::vector<T> dynamic_update_slice(std::vector<T> x, std::vector<T> u,
                                    std::vector<int64_t> startIndex) {
  std::vector<T> result(x);

  auto clamp = [](size_t value, size_t minValue, size_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  size_t startIndex_ = startIndex[0];
  startIndex_ = clamp(startIndex_, 0, InputShape - UpdateShape);

  for (size_t i = 0; i < UpdateShape; i++) {
    result[startIndex_ + i] = u[i];
  }
  return result;
}

// Overload for 2d case
template <typename T, size_t InputShapeX, size_t InputShapeY,
          size_t UpdateShapeX, size_t UpdateShapeY>
std::vector<T> dynamic_update_slice(std::vector<T> x, std::vector<T> u,
                                    std::vector<int64_t> startIndexX,
                                    std::vector<int64_t> startIndexY) {
  std::vector<T> result(x);

  auto clamp = [](size_t value, size_t minValue, size_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  size_t startIndexX_ = startIndexX[0];
  size_t startIndexY_ = startIndexY[0];
  startIndexX_ = clamp(startIndexX_, 0, InputShapeX - UpdateShapeX);
  startIndexY_ = clamp(startIndexY_, 0, InputShapeY - UpdateShapeY);

  for (size_t i = 0; i < UpdateShapeX; i++) {
    for (size_t j = 0; j < UpdateShapeY; j++) {
      result[(startIndexX_ + i) * InputShapeY + (startIndexY_ + j)] =
          u[i * UpdateShapeY + j];
    }
  }
  return result;
}

// ReshapeOp
// This needs to be changed if tensor rank/shape get modelled in the translation
template <typename T>
inline std::vector<T> reshape(std::vector<T> x) {
  return std::vector<T>(x);
}

// SelectOp
template <typename T>
inline std::vector<T> select(std::vector<bool> s, std::vector<T> x,
                             std::vector<T> y) {
  std::vector<T> z(x.size());
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = s[i] ? x[i] : y[i];
  }
  return z;
}

// RngUniformOp
template <typename T>
using IsIntegral =
    typename std::enable_if<std::is_integral<T>::value, bool>::type;
template <typename T>
using IsFloatingPoint =
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type;

// integer types
template <typename T, IsIntegral<T> = true>
std::vector<T> rng_uniform(T low, T high, std::vector<int64_t> shape) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<int64_t>());

  std::random_device rd;
  std::mt19937 gen(rd());
  // high value is exclusive in xla but inclusive in cpp
  // see https://www.tensorflow.org/xla/operation_semantics?hl=en#rnguniform and
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
  std::uniform_int_distribution<T> distribution(low, high - 1);
  std::vector<T> result(n);
  for (size_t i = 0; i < n; i++) {
    result[i] = distribution(gen);
  }
  return result;
}

// floating point types
template <typename T, IsFloatingPoint<T> = true>
std::vector<T> rng_uniform(T low, T high, std::vector<int64_t> shape) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<int64_t>());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> distribution(low, high);
  std::vector<T> result(n);
  for (size_t i = 0; i < n; i++) {
    result[i] = distribution(gen);
  }
  return result;
}

// RngBitGeneratorOp
template <typename T, int32_t Algorithm, int64_t N>
std::tuple<std::vector<uint64_t>, std::vector<T>>
rng_bit_generator(std::vector<uint64_t> state) {
  // TODO implement correct algorithm; starting point would be
  // https://github.com/tensorflow/tensorflow/blob/6f59650012f8904745dffaba540afc794c6613be/tensorflow/compiler/xla/service/rng_bit_generator_expander.cc#L56
  std::vector<uint64_t> newState(state);
  std::vector<int64_t> shape{N};

  T min = std::numeric_limits<T>::min();
  T max = std::numeric_limits<T>::max();
  std::vector<T> resultVector = rng_uniform<T>(min, max, shape);

  return std::make_tuple(newState, resultVector);
}

} // namespace mhlo

#endif // EMITC_MHLO_H
