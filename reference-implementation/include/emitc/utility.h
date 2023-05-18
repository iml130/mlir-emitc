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

// This file defines the utility functions used in EmitC.

#ifndef EMITC_UTILITY_H
#define EMITC_UTILITY_H

#include <array>
#include <cassert>
#include <cstddef>

namespace emitc {
namespace utility {
template <size_t... Shape>
static constexpr size_t size() {
  constexpr std::array<size_t, sizeof...(Shape)> s = {Shape...};

  size_t result = 1;
  for (size_t i = 0; i < sizeof...(Shape); ++i) {
    result *= s[i];
  }
  return result;
}

template <size_t... Shape>
static constexpr std::array<size_t, sizeof...(Shape)> strides() {
  std::array<size_t, sizeof...(Shape)> result = {};
  constexpr std::array<size_t, sizeof...(Shape)> s = {Shape...};

  if (sizeof...(Shape) == 0) {
    return result;
  }

  result[sizeof...(Shape) - 1] = 1;

  for (size_t i = sizeof...(Shape) - 1; i > 0; i--) {
    result[i - 1] = result[i] * s[i];
  }

  return result;
}

template <size_t... Shape>
constexpr size_t ravel_index(std::array<size_t, sizeof...(Shape)> indices) {
  std::array<size_t, sizeof...(Shape)> shape = {Shape...};

  for (size_t i = 0; i < sizeof...(Shape); ++i) {
    assert(indices[i] < shape[i]);
  }

  std::array<size_t, sizeof...(Shape)> s = strides<Shape...>();

  size_t result = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    result += indices[i] * s[i];
  }

  return result;
}

template <size_t... Shape, typename... Indices>
constexpr size_t ravel_index(Indices... indices) {
  static_assert(sizeof...(Indices) == sizeof...(Shape),
                "Incorrect number of arguments");
  return ravel_index<Shape...>({static_cast<size_t>(indices)...});
}

template <size_t... Shape>
constexpr std::array<size_t, sizeof...(Shape)> unravel_index(size_t index) {
  assert(index < size<Shape...>());

  std::array<size_t, sizeof...(Shape)> s = strides<Shape...>();

  std::array<size_t, sizeof...(Shape)> result = {};
  for (size_t i = 0; i < sizeof...(Shape); ++i) {
    result[i] = index / s[i];
    index = index % s[i];
  }

  return result;
}

} // namespace utility
} // namespace emitc

#endif // EMITC_UTILITY_H
