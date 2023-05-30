// Copyright Fraunhofer-Gesellschaft zur Förderung der angewandten
//           Forschung e.V.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file defines functions used by EmitC.

#ifndef EMITC_TENSOR_H
#define EMITC_TENSOR_H

#include <algorithm>

#include "emitc/types.h"

namespace emitc {
namespace tensor {

// ExtractOp
template <typename T, size_t... Shape, typename... Indices>
inline T extract(Tensor<T, Shape...> x, Indices... indices) {
  return x(indices...);
}

// SplatOp
template <typename Dest, typename Src, IsScalar<Src> = true>
inline Dest splat(Src x) {
  Dest z;

  std::fill(z.begin(), z.end(), x);

  return z;
}

} // namespace tensor
} // namespace emitc

#endif // EMITC_TENSOR_H
