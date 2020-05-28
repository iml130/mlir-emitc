//===- EmitCDialect.h - MLIR Dialect to EmitC -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the EmitC in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_EMITCDIALECT_H_
#define MLIR_DIALECT_EMITC_EMITCDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace emitc {

#define GET_OP_CLASSES
#include "mlir/Dialect/EmitC/EmitC.h.inc"

#include "mlir/Dialect/EmitC/EmitCDialect.h.inc"

} // namespace emitc
} // namespace mlir

#endif // MLIR_DIALECT_EMITC_EMITCDIALECT_H_
