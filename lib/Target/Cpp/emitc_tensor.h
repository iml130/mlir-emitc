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

#include <cassert>
#include <cstddef>
#include <vector>

template <typename T, size_t SIZE>
class Tensor {
public:
  using ElementType = T;
  using IteratorType = typename std::vector<T>::iterator;

  Tensor() : data(SIZE) {}

  Tensor(std::initializer_list<T> data) : data(data) {
    assert(data.size() == SIZE);
  }

  IteratorType begin() { return data.begin(); }

  IteratorType end() { return data.end(); }

  // Index into the flat data buffer.
  T &operator[](size_t x) { return data[x]; }

  std::vector<T> data;
  static const size_t size;
};

template <typename T>
class Tensor0D : public Tensor<T, 1> {
public:
  Tensor0D() : Tensor<T, 1>() {}

  Tensor0D(std::initializer_list<T> data) : Tensor<T, 1>(data) {}

  T &operator()() { return this->data.at(0); }
};

template <typename T, size_t DimX>
class Tensor1D : public Tensor<T, DimX> {
public:
  Tensor1D() : Tensor<T, DimX>() {}

  Tensor1D(std::initializer_list<T> data) : Tensor<T, DimX>(data) {}

  T &operator()(size_t x) { return this->data.at(x); }

  static const size_t dimX;
};

template <typename T, size_t DimX, size_t DimY>
class Tensor2D : public Tensor<T, DimX * DimY> {
public:
  Tensor2D() : Tensor<T, DimX * DimY>() {}

  Tensor2D(std::initializer_list<T> data) : Tensor<T, DimX * DimY>(data) {}

  T &operator()(size_t x, size_t y) { return this->data.at(x * DimY + y); }

  static const size_t dimX;
  static const size_t dimY;
};

template <typename T, size_t SIZE>
const size_t Tensor<T, SIZE>::size = SIZE;

template <typename T, size_t DimX>
const size_t Tensor1D<T, DimX>::dimX = DimX;

template <typename T, size_t DimX, size_t DimY>
const size_t Tensor2D<T, DimX, DimY>::dimX = DimX;

template <typename T, size_t DimX, size_t DimY>
const size_t Tensor2D<T, DimX, DimY>::dimY = DimY;

template <typename T>
using is_scalar = std::is_arithmetic<T>;

template <typename T>
struct is_tensor_1d : std::false_type {};

template <typename T, size_t DimX>
struct is_tensor_1d<Tensor1D<T, DimX>> : std::true_type {};

template <typename T>
struct is_tensor_2d : std::false_type {};

template <typename T, size_t DimX, size_t DimY>
struct is_tensor_2d<Tensor2D<T, DimX, DimY>> : std::true_type {};

template <typename T, typename Unused = void>
struct is_tensor : std::false_type {};

template <typename T>
struct is_tensor<T,
                 typename std::enable_if<std::is_base_of<
                     Tensor<typename T::ElementType, T::size>, T>::value>::type>
    : std::true_type {};

template <typename T>
using IsScalar = typename std::enable_if<std::is_scalar<T>::value, bool>::type;

template <typename T>
using IsTensor = typename std::enable_if<is_tensor<T>::value, bool>::type;

template <typename T>
struct get_element_type {
  using type = T;
};

template <typename T, size_t DimX>
struct get_element_type<Tensor1D<T, DimX>> {
  using type = T;
};

template <typename T, size_t DimX, size_t DimY>
struct get_element_type<Tensor2D<T, DimX, DimY>> {
  using type = T;
};
