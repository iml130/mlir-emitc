//===- InitDialect.h - EmitC Dialect Registration----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_INITDIALECT_H
#define EMITC_INITDIALECT_H

#include "emitc/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

// Add all the MLIR dialects to the provided registry.
inline void registerEmitCDialect(DialectRegistry &registry) {
  registry.insert<emitc::EmitCDialect>();
}

} // namespace mlir

#endif // EMITC_INITDIALECT_H
