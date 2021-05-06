//===- TranslateToCRegistration.cpp - Register translation -------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "emitc/Target/Cpp.h"
#include "emitc/Target/TranslationFlags.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"

using namespace mlir;

static LogicalResult MlirToCTranslateFunction(ModuleOp module,
                                              llvm::raw_ostream &output) {
  return emitc::TranslateToC(*module.getOperation(),
                             emitc::getTargetOptionsFromFlags(), output,
                             /*trailingSemiColon=*/false);
}

namespace mlir {
void registerMlirToCTranslation() {
  TranslateFromMLIRRegistration reg(
      "mlir-to-c", MlirToCTranslateFunction, [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<emitc::EmitCDialect,
                        StandardOpsDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}
} // namespace mlir
