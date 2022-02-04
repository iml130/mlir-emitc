//===- Passes.td - EmitC pass definition file --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_DIALECT_EMITC_CONVERSION_PASSES
#define EMITC_DIALECT_EMITC_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMHLORegionOpsToEmitC : Pass<"convert-mhlo-region-ops-to-emitc", "ModuleOp"> {
  let summary = "Convert MHLO operations containing regions to EmitC dialect.";
  let constructor = "createConvertMhloRegionOpsToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertMHLOToEmitC : Pass<"convert-mhlo-to-emitc", "FuncOp"> {
  let summary = "Convert from MHLO dialect to EmitC dialect.";
  let constructor = "createConvertMhloToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertArithToEmitC : Pass<"convert-arith-to-emitc", "FuncOp"> {
  let summary = "Convert arith dialect to EmitC dialect, replacing IndexCastOp.";
  let constructor = "createConvertArithToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertTensorToEmitC : Pass<"convert-tensor-to-emitc", "FuncOp"> {
  let summary = "Convert tensor dialect to EmitC dialect, replacing ExtractOp.";
  let constructor = "createConvertTensorToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertTosaToEmitC : Pass<"convert-tosa-to-emitc", "FuncOp"> {
  let summary = "Convert TOSA dialect to EmitC dialect.";
  let constructor = "createConvertTosaToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

#endif // EMITC_DIALECT_EMITC_CONVERSION_PASSES
