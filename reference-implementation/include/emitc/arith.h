// Copyright Fraunhofer-Gesellschaft zur Förderung der angewandten
//           Forschung e.V.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file defines functions used by EmitC.

#ifndef EMITC_ARITH_H
#define EMITC_ARITH_H

#include "emitc/types.h"

namespace emitc {
namespace arith {

// IndexCastOp
template <typename Dest, typename Src>
inline Dest index_cast(Src x) {
  using ET_Dest = typename get_element_type<Dest>::type;
  using ET_Src = typename get_element_type<Src>::type;

  auto cast = [](ET_Src value) { return static_cast<ET_Dest>(value); };

  return unary<Dest, Src, UnaryFuncType<ET_Dest, ET_Src>>(x, cast);
}

} // namespace arith
} // namespace emitc

#endif // EMITC_ARITH_H
