//===- InitPasses.h - EmitC Pass Registration -------------------*- C++ -*-===//
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

#ifndef MLIR_INITPASSES_H
#define MLIR_INITPASSES_H

#include "emitc/Dialect/EmitC/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

#include <cstdlib>

namespace mlir {

// This function may be called to register the MLIR passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
inline void registerEmitCPasses() {

  // TODO: Register Passes

}

} // namespace mlir

#endif // MLIR_INITPASSES_H
