//===- Passes.h - Conversion Passes ------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_CONVERSION_PASSES_H
#define EMITC_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace emitc {

#ifdef EMITC_BUILD_HLO
std::unique_ptr<OperationPass<FuncOp>> createConvertMHLOToStandardPass();

#define GEN_PASS_REGISTRATION
#include "emitc/Conversion/Passes.h.inc"

#endif // EMITC_BUILD_HLO

} // namespace emitc
} // namespace mlir

#endif // EMITC_CONVERSION_PASSES_H
