//===- Passes.h - EmitC Transform Passes ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines transform passes owned by the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_DIALECT_EMITC_TRANSFORMS_PASSES_H
#define EMITC_DIALECT_EMITC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace emitc {

#ifdef EMITC_BUILD_HLO
std::unique_ptr<OperationPass<ModuleOp>> createInsertEmitCMHLOIncludePass();
#endif // EMITC_BUILD_HLO
std::unique_ptr<OperationPass<ModuleOp>> createInsertEmitCArithIncludePass();
std::unique_ptr<OperationPass<ModuleOp>> createInsertEmitCTensorIncludePass();
std::unique_ptr<OperationPass<ModuleOp>> createInsertEmitCTosaIncludePass();

#define GEN_PASS_REGISTRATION
#include "emitc/Dialect/EmitC/Transforms/Passes.h.inc"

} // namespace emitc
} // namespace mlir

#endif // EMITC_DIALECT_EMITC_TRANSFORMS_PASSES_H
