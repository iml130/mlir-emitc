//===- TranslateToCppRegistration.cpp - Register for mlir-translate  ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Module.h"
#include "mlir/Target/Cpp.h"
#include "mlir/Translation.h"

using namespace mlir;

static LogicalResult MlirToCppTranslateFunction(ModuleOp module,
                                                llvm::raw_ostream &output) {
  return emitc::TranslateToCpp(*module.getOperation(), output,
                               /*trailingSemiColon=*/false);
}

static TranslateFromMLIRRegistration reg("mlir-to-cpp",
                                         MlirToCppTranslateFunction);
