//===- Passes.h - EmitC Passes ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines passes owned by the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_PASSES_H
#define MLIR_DIALECT_EMITC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace emitc {

std::unique_ptr<OperationPass<FuncOp>> createConvertMhloToEmitcPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertScfToEmitcPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertStdToEmitcPass();

} // namespace emitc
} // namespace mlir

#endif // MLIR_DIALECT_EMITC_PASSES_H
