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

// This file defines tensor classes used by EmitC

#ifndef EMITC_TENSOR_H
#define EMITC_TENSOR_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace {
template <size_t N>
constexpr size_t sum(std::array<size_t, N> arr) {
  size_t result = 0;

  for (size_t i = 0; i < arr.size(); i++) {
    result += arr[i];
  }
  return result;
}

template <size_t N>
constexpr size_t first(std::array<size_t, N> arr) {
  return arr[0];
}

template <size_t N>
constexpr bool all_same(std::array<size_t, N> arr) {
  if (arr.size() == 0) {
    return true;
  }

  size_t first = arr[0];

  for (size_t i = 1; i < arr.size(); i++) {
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
} // namespace

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

  static constexpr size_t size() {
    constexpr std::array<size_t, rank()> s = {Shape...};

    size_t result = 1;
    for (size_t i = 0; i < rank(); i++) {
      result *= s[i];
    }
    return result;
  }

  static constexpr std::array<size_t, rank()> strides() {
    std::array<size_t, rank()> result;
    constexpr std::array<size_t, rank()> s = {Shape...};

    result[rank() - 1] = 1;
    size_t i = rank() - 2;

    do {
      result[i] = result[i + 1] * s[i + 1];
    } while (i-- > 0);

    return result;
  }

  iterator begin() { return data.begin(); }

  const_iterator begin() const { return data.begin(); }

  iterator end() { return data.end(); }

  const_iterator end() const { return data.end(); }

  // Index into the flat data buffer.
  reference operator[](size_t x) {
    assert(0 <= x && x < size());
    return data[x];
  }

  template <typename... Indices,
            typename =
                std::enable_if<conjunction_v<std::is_same<size_t, Indices>...>>>
  reference operator()(Indices... indices) {
    size_t index = ravel_index(indices...);
    assert(0 <= index && index < size());
    return data[index];
  }

private:
  template <typename... Indices, size_t Index = 0,
            typename =
                std::enable_if<conjunction_v<std::is_same<size_t, Indices>...>>>
  constexpr size_t ravel_index(size_t index, Indices... indices) {
    return index * strides()[Index] + ravel_index<Index + 1>(indices...);
  }

  template <size_t Unused = 0>
  constexpr size_t ravel_index(size_t index) {
    return index;
  }

  constexpr size_t ravel_index() { return 0; }

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
using IsTensor = typename std::enable_if_t<is_tensor<T>::value, bool>;

template <typename T>
struct get_element_type {
  using type = T;
};

template <typename T, size_t... Shape>
struct get_element_type<Tensor<T, Shape...>> {
  using type = T;
};

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
inline Dest unary(Src x, UnaryOp &&op) {
  return op(x);
}

template <typename Dest, typename Src, typename UnaryOp, IsTensor<Src> = true>
inline Dest unary(Src x, UnaryOp &&op) {
  Dest z;
  std::transform(x.begin(), x.end(), z.begin(), op);
  return z;
}

template <typename Dest, typename SrcLeft, typename SrcRight, typename BinaryOp,
          IsScalar<SrcLeft> = true, IsScalar<SrcRight> = true>
inline Dest binary(SrcLeft x, SrcRight y, BinaryOp &&op) {
  return op(x, y);
}

template <typename Dest, typename SrcLeft, typename SrcRight, typename BinaryOp,
          IsTensor<SrcLeft> = true, IsTensor<SrcRight> = true>
inline Dest binary(SrcLeft x, SrcRight y, BinaryOp &&op) {
  Dest z;
  std::transform(x.begin(), x.end(), y.begin(), z.begin(), op);
  return z;
}

template <size_t Dim, typename T, typename... Ts>
struct concat {};

template <size_t Dim, typename T, size_t... Xs>
struct concat<Dim, T, Tensor1D<T, Xs>...> {
  static_assert(0 <= Dim && Dim < 1, "Dimension index out of bounds");
  using type = Tensor1D<T, sum<Xs...>()>;
};

template <typename T, size_t Dim, size_t... Xs, size_t... Ys>
struct concat<Dim, T, Tensor2D<T, Xs, Ys>...> {
  static_assert(0 <= Dim && Dim < 2, "Dimension index out of bounds");
  static_assert((Dim == 0 && all_same({Ys...})) ||
                    (Dim == 1 && all_same({Xs...})),
                "All dimensions except for the dimension index must match");
  using type =
      typename std::conditional_t<Dim == 0,
                                  Tensor2D<T, sum({Xs...}), first({Ys...})>,
                                  Tensor2D<T, first({Xs...}), sum({Ys...})>>;
};

#endif // EMITC_TENSOR_H
