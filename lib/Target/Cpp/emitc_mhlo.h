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
#include <vector>

namespace mhlo {

/// Functions for MHLO unary elementwise ops
// AbsOp
template <typename T1, typename T2>
inline T1 abs(T2 x) {
  return std::abs(x);
}

template <typename T>
inline std::vector<T> abs(std::vector<T> x) {
  std::vector<T> z(x);
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = std::abs(x[i]);
  }
  return z;
}

// AbsOp supports complex to real.
template <typename T>
inline std::vector<T> abs(std::vector<std::complex<T>> x) {
  std::vector<T> z;
  z.reserve(x.size());
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = std::abs(x[i]);
  }
  return z;
}

// ConvertOp
// TODO: Implement!

// CosOp
template <typename T>
inline T cos(T x) {
  return std::cos(x);
}

template <typename T>
inline std::vector<T> cos(std::vector<T> x) {
  std::vector<T> z(x);
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = std::cos(x[i]);
  }
  return z;
}

// ExpOp
// TODO: Implement!
// `e^(operand)` element-wise

// IsFiniteOp
// TODO: Implement!

// LogOp
// TODO: Implement!

// NegOp
// TODO: Implement!

/// Functions for MHLO binary elementwise ops.
// AddOp
template <typename T>
inline T add(T x, T y) {
  return std::plus<>{}(x, y);
}

template <typename T>
inline std::vector<T> add(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  std::transform(x.begin(), x.end(), y.begin(), z.begin(), std::plus<>());
  return z;
}

// DivOp
template <typename T>
inline T div(T x, T y) {
  return std::divides<>{}(x, y);
}

template <typename T>
inline std::vector<T> div(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  std::transform(x.begin(), x.end(), y.begin(), z.begin(), std::divides<>());
  return z;
}

// MaxOp
template <typename T>
inline T max(T x, T y) {
  return std::max(x, y);
}

template <typename T>
inline std::vector<T> max(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  std::transform(x.begin(), x.end(), y.begin(), z.begin(),
                 [](auto a, auto b) { return std::max(a, b); });
  return z;
}

// MinOp
template <typename T>
inline T min(T x, T y) {
  return std::min(x, y);
}

template <typename T>
inline std::vector<T> min(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  std::transform(x.begin(), x.end(), y.begin(), z.begin(),
                 [](auto a, auto b) { return std::min(a, b); });
  return z;
}

// MulOp
template <typename T>
inline T mul(T x, T y) {
  return std::multiplies<>{}(x, y);
}

template <typename T>
inline std::vector<T> mul(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  std::transform(x.begin(), x.end(), y.begin(), z.begin(), std::multiplies<>());
  return z;
}

// PowOp
template <typename T>
inline T pow(T x, T y) {
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
template <typename T>
inline T sub(T x, T y) {
  return std::minus<>{}(x, y);
}

template <typename T>
inline std::vector<T> sub(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  std::transform(x.begin(), x.end(), y.begin(), z.begin(), std::minus<>());
  return z;
}

/// Functions for MHLO binary logical elementwise ops.
// OrOp
// TODO: Implement!

// XorOp
// TODO: Implement!

/// Functions for other MHLO ops.
// ConcatenateOp
template <typename T>
inline std::vector<T> concatenate(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  z.insert(z.end(), y.begin(), y.end());
  return z;
}

} // namespace mhlo

#endif // EMITC_MHLO_H
