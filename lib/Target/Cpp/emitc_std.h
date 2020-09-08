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

#ifndef EMITC_STD_H
#define EMITC_STD_H

#include <vector>

#include "emitc_tensor.h"

namespace standard {

// ExtractElementOp
template <typename T>
inline T extract_element(Tensor0D<T> x) {
  return x();
}

template <typename T, size_t DimX>
inline T extract_element(Tensor1D<T, DimX> x, size_t indexX) {
  return x(indexX);
}

template <typename T, size_t DimX, size_t DimY>
inline T extract_element(Tensor2D<T, DimX, DimY> x, size_t indexX, size_t indexY) {
  return x(indexX, indexY);
}

// IndexCastOp
template <typename T1, typename T2>
inline T1 index_cast(T2 x) {
  return static_cast<T1>(x);
}

template <typename T1, typename T2>
inline Tensor0D<T1> index_cast(Tensor0D<T2> x) {
  Tensor0D<T1> z;
  z[0] = static_cast<T1>(x[0]);
  return z;
}

template <typename T1, typename T2, size_t DimX>
inline Tensor1D<T1, DimX> index_cast(Tensor1D<T2, DimX> x) {
  Tensor1D<T1, DimX> z;
  for (size_t i = 0; i < x.size; i++) {
    z[i] = static_cast<T1>(x[i]);
  }
  return z;
}

template <typename T1, typename T2, size_t DimX, size_t DimY>
inline Tensor2D<T1, DimX, DimY> index_cast(Tensor2D<T2, DimX, DimY> x) {
  Tensor2D<T1, DimX, DimY> z;
  for (size_t i = 0; i < x.size; i++) {
    z[i] = static_cast<T1>(x[i]);
  }
  return z;
}

} // namespace standard

#endif // EMITC_STD_H
