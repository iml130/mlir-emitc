//===- Passes.td - EmitC pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_DIALECT_EMITC_PASSES
#define EMITC_DIALECT_EMITC_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMHLOToEmitCPass : Pass<"convert-mhlo-to-emitc", "ModuleOp"> {
  let summary = "Convert from MHLO dialect to EmitC dialect.";
  let constructor = "createConvertMhloToEmitcPass()";
}

def ConvertScfToEmitCPass : Pass<"convert-scf-to-emitc", "FuncOp"> {
  let summary = "Convert SCF dialect to EmitC dialect, replacing IfOp and ForOp.";
  let constructor = "createConvertScfToEmitcPass()";
}

def ConvertStdToEmitCPass : Pass<"convert-std-to-emitc", "FuncOp"> {
  let summary = "Convert std dialect to EmitC dialect, replacing IndexCastOp.";
  let constructor = "createConvertStdToEmitcPass()";
}

// TODO: This could be moved to separate file.
def ConvertMHLOToStandardPass : Pass<"convert-mhlo-const-to-std", "FuncOp"> {
  let summary = "Converting MHLO constant to std.";
  let constructor = "mhlo::createConvertMHLOToStandardPass()";
  let dependentDialects = ["StandardOpsDialect"];
}

#endif // EMITC_DIALECT_EMITC_PASSES
