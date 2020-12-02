//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_PASSDETAIL_H
#define CONVERSION_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class StandardOpsDialect;

namespace emitc {

class EmitCDialect;

#define GEN_PASS_CLASSES
#include "emitc/Conversion/Passes.h.inc"

} // namespace emitc
} // namespace mlir

#endif // CONVERSION_PASSDETAIL_H
