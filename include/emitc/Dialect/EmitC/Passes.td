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

def ConvertMHLORegionOpsToEmitC : Pass<"convert-mhlo-region-ops-to-emitc", "ModuleOp"> {
  let summary = "Convert MHLO operations containing regions to EmitC dialect.";
  let constructor = "createConvertMhloRegionOpsToEmitcPass()";
}

def ConvertMHLOToEmitC : FunctionPass<"convert-mhlo-to-emitc"> {
  let summary = "Convert from MHLO dialect to EmitC dialect.";
  let constructor = "createConvertMhloToEmitcPass()";
}

def ConvertSCFToEmitC : Pass<"convert-scf-to-emitc", "FuncOp"> {
  let summary = "Convert SCF dialect to EmitC dialect, replacing IfOp and ForOp.";
  let constructor = "createConvertScfToEmitcPass()";
}

def ConvertStdToEmitC : Pass<"convert-std-to-emitc", "FuncOp"> {
  let summary = "Convert std dialect to EmitC dialect, replacing IndexCastOp.";
  let constructor = "createConvertStdToEmitcPass()";
}

#endif // EMITC_DIALECT_EMITC_PASSES
