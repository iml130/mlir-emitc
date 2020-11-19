//===- Passes.td - Pass definition file --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_CONVERSION_PASSES
#define EMITC_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMHLOToStandard : Pass<"convert-mhlo-const-to-std", "FuncOp"> {
  let summary = "Converting MHLO constant to std.";
  let constructor = "mhlo::createConvertMHLOToStandardPass()";
  let dependentDialects = ["StandardOpsDialect"];
}

#endif // EMITC_CONVERSION_PASSES
