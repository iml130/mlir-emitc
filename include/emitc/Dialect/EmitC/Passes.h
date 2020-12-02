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

#ifndef EMITC_DIALECT_EMITC_PASSES_H
#define EMITC_DIALECT_EMITC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace emitc {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMhloRegionOpsToEmitcPass();
std::unique_ptr<FunctionPass> createConvertMhloToEmitcPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertScfToEmitcPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertStdToEmitcPass();

#define GEN_PASS_REGISTRATION
#include "emitc/Dialect/EmitC/Passes.h.inc"

} // namespace emitc
} // namespace mlir

#endif // EMITC_DIALECT_EMITC_PASSES_H
