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

#ifndef MLIR_INITDIALECT_H
#define MLIR_INITDIALECT_H

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

// Add all the MLIR dialects to the provided registry.
inline void registerEmitCDialect(DialectRegistry &registry) {
  registry.insert<emitc::EmitCDialect>();
}

// This function should be called before creating any MLIRContext if one expect
// all the possible dialects to be made available to the context automatically.
inline void registerEmitCDialect() {
  static bool initOnce =
      ([]() { registerEmitCDialect(getGlobalDialectRegistry()); }(), true);
  (void)initOnce;
}
} // namespace mlir

#endif // MLIR_INITDIALECT_H
