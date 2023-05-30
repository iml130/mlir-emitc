//===- Passes.h - EmitC Conversion Passes -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines conversion passes owned by the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_CONVERSION_PASSES_H
#define EMITC_CONVERSION_PASSES_H

#include "emitc/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "emitc/Conversion/StablehloToEmitC/StablehloToEmitC.h"
#include "emitc/Conversion/TensorToEmitC/TensorToEmitC.h"
#include "emitc/Conversion/TosaToEmitC/TosaToEmitC.h"

namespace mlir {
namespace emitc {

#define GEN_PASS_REGISTRATION
#include "emitc/Conversion/Passes.h.inc"

} // namespace emitc
} // namespace mlir

#endif // EMITC_CONVERSION_PASSES_H
