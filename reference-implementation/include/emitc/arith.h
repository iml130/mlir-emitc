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
