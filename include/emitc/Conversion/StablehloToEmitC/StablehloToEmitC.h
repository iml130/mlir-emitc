//===- StablehloToEmitC.h - Convert StableHLO to EmitC dialect --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_CONVERSION_STABLEHLOTOEMITC_H
#define EMITC_CONVERSION_STABLEHLOTOEMITC_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace func {
class FuncOp;
} // namespace func

namespace emitc {

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertStablehloToEmitCPass();

} // namespace emitc
} // namespace mlir

#endif // EMITC_CONVERSION_STABLEHLOTOEMITC_H
