//===- TranslateRegistration.cpp - Register translation ----------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/IR/EmitC.h"
#include "emitc/Target/Cpp/Cpp.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

void registerToCppTranslation() {
  TranslateFromMLIRRegistration reg(
      "mlir-to-cpp",
      [](ModuleOp module, llvm::raw_ostream &output) {
        return emitc::TranslateToCpp(*module.getOperation(), output,
                                     /*forwardDeclareVariables=*/false,
                                     /*trailingSemiColon=*/false);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<emitc::EmitCDialect,
                        StandardOpsDialect,
                        scf::SCFDialect>();
        // clang-format on
      });

  TranslateFromMLIRRegistration regForwardDeclared(
      "mlir-to-cpp-forward-declared",
      [](ModuleOp module, llvm::raw_ostream &output) {
        return emitc::TranslateToCpp(*module.getOperation(), output,
                                     /*forwardDeclareVariables=*/true,
                                     /*trailingSemiColon=*/false);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<emitc::EmitCDialect,
                        StandardOpsDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}

//===----------------------------------------------------------------------===//
// C registration
//===----------------------------------------------------------------------===//

void registerToCTranslation() {
  TranslateFromMLIRRegistration reg(
      "mlir-to-c",
      [](ModuleOp module, llvm::raw_ostream &output) {
        return emitc::TranslateToC(*module.getOperation(), output,
                                   /*forwardDeclareVariables=*/false,
                                   /*trailingSemiColon=*/false);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<emitc::EmitCDialect,
                        StandardOpsDialect,
                        scf::SCFDialect>();
        // clang-format on
      });

  TranslateFromMLIRRegistration regForwardDeclared(
      "mlir-to-c-forward-declared",
      [](ModuleOp module, llvm::raw_ostream &output) {
        return emitc::TranslateToC(*module.getOperation(), output,
                                   /*forwardDeclareVariables=*/true,
                                   /*trailingSemiColon=*/false);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<emitc::EmitCDialect,
                        StandardOpsDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}

} // namespace mlir
