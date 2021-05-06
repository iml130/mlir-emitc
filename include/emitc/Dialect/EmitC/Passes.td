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
  let constructor = "createConvertMhloRegionOpsToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertMHLOToEmitC : FunctionPass<"convert-mhlo-to-emitc"> {
  let summary = "Convert from MHLO dialect to EmitC dialect.";
  let constructor = "createConvertMhloToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertStdToEmitC : FunctionPass<"convert-std-to-emitc"> {
  let summary = "Convert std dialect to EmitC dialect, replacing IndexCastOp and SplatOp.";
  let constructor = "createConvertStdToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertTensorToEmitC : FunctionPass<"convert-tensor-to-emitc"> {
  let summary = "Convert tensor dialect to EmitC dialect, replacing ExtractOp.";
  let constructor = "createConvertTensorToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertTosaToEmitC : FunctionPass<"convert-tosa-to-emitc"> {
  let summary = "Convert TOSA dialect to EmitC dialect.";
  let constructor = "createConvertTosaToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

#endif // EMITC_DIALECT_EMITC_PASSES
