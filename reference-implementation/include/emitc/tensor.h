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

// This file defines functions used by EmitC.

#ifndef EMITC_TENSOR_H
#define EMITC_TENSOR_H

#include "emitc/types.h"

namespace emitc {
namespace tensor {

// ExtractOp
template <typename T, size_t... Shape, typename... Indices>
inline T extract(Tensor<T, Shape...> x, Indices... indices) {
  return x(indices...);
}

} // namespace tensor
} // namespace emitc

#endif // EMITC_TENSOR_H
