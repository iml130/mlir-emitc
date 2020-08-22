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

#include <cstdint>
#include <vector>

namespace standard {

// ExtractElementOp
// Special case for 0-dim tensors;
template <typename T>
inline T extract_element(std::vector<T> x) {
  return x[0];
}

template <typename T>
inline T extract_element(std::vector<T> x, size_t index) {
  return x[index];
}

// IndexCastOp
template <typename T1, typename T2>
inline T1 index_cast(T2 x) {
  return static_cast<T1>(x);
}

template <typename T1, typename T2>
inline std::vector<T1> index_cast(std::vector<T2> x) {
  std::vector<T1> z(x.size());
  for (size_t i = 0; i < z.size(); i++) {
    z[i] = static_cast<T1>(x[i]);
  }
  return z;
}

} // namespace standard

#endif // EMITC_STD_H
