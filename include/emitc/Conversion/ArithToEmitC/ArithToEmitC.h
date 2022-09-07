//===- ArithToEmitC.h - Convert Arith to EmitC dialect ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_CONVERSION_ARITHTOEMITC_H
#define EMITC_CONVERSION_ARITHTOEMITC_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace emitc {

std::unique_ptr<OperationPass<func::FuncOp>> createConvertArithToEmitCPass();

} // namespace emitc
} // namespace mlir

#endif // EMITC_CONVERSION_ARITHTOEMITC_H
