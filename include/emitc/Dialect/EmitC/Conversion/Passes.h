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

#ifndef EMITC_DIALECT_EMITC_CONVERSION_PASSES_H
#define EMITC_DIALECT_EMITC_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace emitc {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMhloRegionOpsToEmitCPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertMhloToEmitCPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertArithToEmitCPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertStdToEmitCPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertTensorToEmitCPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertTosaToEmitCPass();

#define GEN_PASS_REGISTRATION
#include "emitc/Dialect/EmitC/Conversion/Passes.h.inc"

} // namespace emitc
} // namespace mlir

#endif // EMITC_DIALECT_EMITC_CONVERSION_PASSES_H
