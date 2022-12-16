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

// This file defines the tensor class used by EmitC.

#ifndef EMITC_TYPES_H
#define EMITC_TYPES_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#include "emitc/utility.h"

namespace detail {
template <size_t N>
constexpr size_t sum(const std::array<size_t, N> arr) {
  size_t result = 0;

  for (size_t i = 0; i < arr.size(); ++i) {
    result += arr[i];
  }
  return result;
}

template <size_t N>
constexpr size_t first(const std::array<size_t, N> arr) {
  static_assert(N > 0, "Cannot get the first element of an empty array");
  return arr[0];
}

template <size_t N>
constexpr bool all_same(const std::array<size_t, N> arr) {
  if (arr.size() == 0) {
    return true;
  }

  size_t first = arr[0];

  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i] != first) {
      return false;
    }
  }
  return true;
}

template <class...>
struct conjunction : std::true_type {};
template <class B1>
struct conjunction<B1> : B1 {};
template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

template <class... B>
constexpr bool conjunction_v = conjunction<B...>::value;

// Template switch case statement.
// case_t => condition/case
template <bool B, typename T>
struct case_t {
  static constexpr bool value = B;
  using type = T;
};

// switch_t => template switch case
template <typename First, typename... Rest>
struct switch_t : std::conditional_t<First::value, First, switch_t<Rest...>> {};

template <typename T>
struct switch_t<T> {
  using type = T;
};

template <bool B, typename T>
struct switch_t<case_t<B, T>> {
  // One statement needs to be true.
  static_assert(B, "None of the supplied conditions evaluate to true.");
  using type = T;
};
} // namespace detail

template <typename T, size_t... Shape>
class Tensor {
public:
  using value_type = T;
  using reference = typename std::vector<T>::reference;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  Tensor() : data(size()) {}

  Tensor(std::initializer_list<T> data) : data(data) {
    assert(data.size() == size());
  }

  static constexpr size_t dim(size_t index) {
    assert(0 <= index && index < rank());
    constexpr std::array<size_t, rank()> s = {Shape...};
    return s[index];
  }

  static constexpr size_t rank() { return sizeof...(Shape); }

  static constexpr std::array<size_t, rank()> shape() { return {Shape...}; }

  static constexpr size_t size() { return emitc::utility::size<Shape...>(); }

  static constexpr std::array<size_t, rank()> strides() {
    return emitc::utility::strides<Shape...>();
  }

  std::vector<std::array<size_t, rank()>>
  window(std::array<size_t, rank()> index, std::array<size_t, rank()> sizes) {
    std::vector<std::vector<size_t>> iotas;
    for (auto &size : sizes) {
      std::vector<size_t> range(size);
      std::iota(range.begin(), range.end(), 0);
      iotas.push_back(range);
    }

    std::vector<std::array<size_t, rank()>> result;

    int resultSize =
        std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>{});
    for (int n = 0; n < resultSize; ++n) {
      std::array<size_t, rank()> u = {};
      div_t q{n, 0};
      for (int i = iotas.size() - 1; 0 <= i; --i) {
        q = div(q.quot, iotas[i].size());
        u[i] = iotas[i][q.rem];
      }

      for (size_t i = 0; i < index.size(); ++i) {
        u[i] += index[i];
      }
      result.push_back(u);
    }

    return result;
  }

  iterator begin() { return data.begin(); }

  const_iterator begin() const { return data.begin(); }

  iterator end() { return data.end(); }

  const_iterator end() const { return data.end(); }

  // Index into the flat data buffer.
  reference operator[](size_t index) {
    assert(0 <= index && index < size());
    return data[index];
  }

  template <typename... Indices,
            typename = std::enable_if<
                detail::conjunction_v<std::is_same<size_t, Indices>...>>>
  reference operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == rank(),
                  "Incorrect number of arguments");
    size_t index = ravel_index({static_cast<size_t>(indices)...});

    assert(index < size());
    return data[index];
  }

  constexpr size_t ravel_index(std::array<size_t, rank()> indices) {
    return emitc::utility::ravel_index<Shape...>(indices);
  }

  constexpr std::array<size_t, rank()> unravel_index(size_t index) {
    return emitc::utility::unravel_index<Shape...>(index);
  }

private:
  std::vector<T> data;
};

template <typename T>
using Tensor0D = Tensor<T>;

template <typename T, size_t Dim0>
using Tensor1D = Tensor<T, Dim0>;

template <typename T, size_t Dim0, size_t Dim1>
using Tensor2D = Tensor<T, Dim0, Dim1>;

template <typename T, size_t Dim0, size_t Dim1, size_t Dim2>
using Tensor3D = Tensor<T, Dim0, Dim1, Dim2>;

template <typename T, size_t Dim0, size_t Dim1, size_t Dim2, size_t Dim3>
using Tensor4D = Tensor<T, Dim0, Dim1, Dim2, Dim3>;

template <typename T>
using is_scalar = std::is_arithmetic<T>;

template <typename T, typename Unused = void>
struct is_tensor : std::false_type {};

template <typename T, size_t... Shape>
struct is_tensor<Tensor<T, Shape...>> : std::true_type {};

template <size_t Dim, typename T, typename Unused = void>
struct is_tensor_of_dim : std::false_type {};

template <size_t Dim, typename T, size_t... Shape>
struct is_tensor_of_dim<Dim, Tensor<T, Shape...>> {
  static constexpr bool value = Tensor<T, Shape...>::rank() == Dim;
};

template <typename T>
using IsScalar = typename std::enable_if_t<std::is_scalar<T>::value, bool>;

template <typename T>
using IsTensor = typename std::enable_if_t<is_tensor<T>::value, bool>;

template <size_t Dim, typename T>
using IsTensorOfDim =
    typename std::enable_if_t<is_tensor_of_dim<Dim, T>::value, bool>;

template <typename T>
struct get_element_type {
  using type = T;
};

template <typename T, size_t... Shape>
struct get_element_type<Tensor<T, Shape...>> {
  using type = T;
};

template <typename T, typename ET>
using IsTensorOfType = std::enable_if_t<
    std::is_same<typename get_element_type<T>::type, ET>::value, bool>;

template <typename Dest, typename Src>
struct replace_element_type {
  using type = Dest;
};

template <typename Dest, typename Src, size_t... Shape>
struct replace_element_type<Dest, Tensor<Src, Shape...>> {
  using type = Tensor<Dest, Shape...>;
};

template <typename Dest, typename Src>
using UnaryFuncType = Dest (*)(Src);

template <typename Dest, typename SrcLeft, typename SrcRight>
using BinaryFuncType = Dest (*)(SrcLeft, SrcRight);

template <typename Dest, typename Src, typename UnaryOp, IsScalar<Src> = true>
inline Dest unary(const Src &x, UnaryOp &&op) {
  return op(x);
}

template <typename Dest, typename Src, typename UnaryOp, IsTensor<Src> = true>
inline Dest unary(const Src &x, UnaryOp &&op) {
  Dest z;
  std::transform(x.begin(), x.end(), z.begin(), op);
  return z;
}

template <typename Dest, typename SrcLeft, typename SrcRight, typename BinaryOp,
          IsScalar<SrcLeft> = true, IsScalar<SrcRight> = true>
inline Dest binary(const SrcLeft &x, const SrcRight &y, BinaryOp &&op) {
  return op(x, y);
}

template <typename Dest, typename SrcLeft, typename SrcRight, typename BinaryOp,
          IsTensor<SrcLeft> = true, IsTensor<SrcRight> = true>
inline Dest binary(const SrcLeft &x, const SrcRight &y, BinaryOp &&op) {
  Dest z;
  std::transform(x.begin(), x.end(), y.begin(), z.begin(), op);
  return z;
}

template <typename Dest, typename SrcA, typename SrcB, typename SrcC,
          typename TernaryOp, IsScalar<SrcA> = true, IsScalar<SrcB> = true,
          IsScalar<SrcC> = true>
inline Dest ternary(const SrcA &a, const SrcB &b, const SrcB &c,
                    TernaryOp &&op) {
  return op(a, b, c);
}

template <typename Dest, typename SrcA, typename SrcB, typename SrcC,
          typename TernaryOp, IsTensor<SrcA> = true, IsTensor<SrcB> = true,
          IsTensor<SrcC> = true>
inline Dest ternary(const SrcA &a, const SrcB &b, const SrcB &c,
                    TernaryOp &&op) {
  Dest d;
  auto first1 = a.begin(), last1 = a.end();
  auto first2 = b.begin(), first3 = c.begin();
  auto result = d.begin();

  while (first1 != last1) {
    *result = op(*first1, *first2, *first3);
    result++;
    first1++;
    first2++;
    first3++;
  }
  return d;
}

template <size_t Dim, typename T, typename... Ts>
struct concat {};

template <size_t Dim, typename T, size_t... Xs>
struct concat<Dim, T, Tensor1D<T, Xs>...> {
  static_assert(0 <= Dim && Dim < 1, "Dimension index out of bounds");
  using type = Tensor1D<T, detail::sum<sizeof...(Xs)>({Xs...})>;
};

template <typename T, size_t Dim, size_t... Xs, size_t... Ys>
struct concat<Dim, T, Tensor2D<T, Xs, Ys>...> {
  static_assert(0 <= Dim && Dim < 2, "Dimension index out of bounds");
  static_assert((Dim == 0 && detail::all_same<sizeof...(Ys)>({Ys...})) ||
                    (Dim == 1 && detail::all_same<sizeof...(Xs)>({Xs...})),
                "All dimensions except for the dimension index must match");
  using type = typename std::conditional_t<
      Dim == 0,
      Tensor2D<T, detail::sum<sizeof...(Xs)>({Xs...}),
               detail::first<sizeof...(Ys)>({Ys...})>,
      Tensor2D<T, detail::first<sizeof...(Xs)>({Xs...}),
               detail::sum<sizeof...(Ys)>({Ys...})>>;
};

template <typename T, size_t Dim, size_t... Xs, size_t... Ys, size_t... Zs>
struct concat<Dim, T, Tensor3D<T, Xs, Ys, Zs>...> {
  static_assert(0 <= Dim && Dim < 3, "Dimension index out of bounds");

  using type = typename detail::switch_t<
      detail::case_t<Dim == 0, Tensor3D<T, detail::sum<sizeof...(Xs)>({Xs...}),
                                        detail::first<sizeof...(Ys)>({Ys...}),
                                        detail::first<sizeof...(Zs)>({Zs...})>>,
      detail::case_t<Dim == 1,
                     Tensor3D<T, detail::first<sizeof...(Xs)>({Xs...}),
                              detail::sum<sizeof...(Ys)>({Ys...}),
                              detail::first<sizeof...(Zs)>({Zs...})>>,
      detail::case_t<Dim == 2,
                     Tensor3D<T, detail::first<sizeof...(Xs)>({Xs...}),
                              detail::first<sizeof...(Ys)>({Ys...}),
                              detail::sum<sizeof...(Zs)>({Zs...})>>>::type;
};

template <typename T, size_t Dim, size_t... D0, size_t... D1, size_t... D2,
          size_t... D3>
struct concat<Dim, T, Tensor4D<T, D0, D1, D2, D3>...> {
  static_assert(0 <= Dim && Dim < 4, "Dimension index out of bounds");

  using type = typename detail::switch_t<
      detail::case_t<Dim == 0, Tensor4D<T, detail::sum<sizeof...(D0)>({D0...}),
                                        detail::first<sizeof...(D1)>({D1...}),
                                        detail::first<sizeof...(D2)>({D2...}),
                                        detail::first<sizeof...(D3)>({D3...})>>,
      detail::case_t<Dim == 1,
                     Tensor4D<T, detail::first<sizeof...(D0)>({D0...}),
                              detail::sum<sizeof...(D1)>({D1...}),
                              detail::first<sizeof...(D2)>({D2...}),
                              detail::first<sizeof...(D3)>({D3...})>>,
      detail::case_t<Dim == 2,
                     Tensor4D<T, detail::first<sizeof...(D0)>({D0...}),
                              detail::first<sizeof...(D1)>({D1...}),
                              detail::sum<sizeof...(D2)>({D2...}),
                              detail::first<sizeof...(D3)>({D3...})>>,
      detail::case_t<Dim == 3,
                     Tensor4D<T, detail::first<sizeof...(D0)>({D0...}),
                              detail::first<sizeof...(D1)>({D1...}),
                              detail::first<sizeof...(D2)>({D2...}),
                              detail::sum<sizeof...(D3)>({D3...})>>>::type;
};
#endif // EMITC_TYPES_H
