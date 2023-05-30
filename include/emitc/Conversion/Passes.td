//===- Passes.td - EmitC pass definition file --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_CONVERSION_PASSES
#define EMITC_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertStablehloRegionOpsToEmitC : Pass<"convert-stablehlo-region-ops-to-emitc", "ModuleOp"> {
  let summary = "Convert StableHLO operations containing regions to EmitC dialect.";
  let constructor = "createConvertStablehloRegionOpsToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertStablehloToEmitC : Pass<"convert-stablehlo-to-emitc", "func::FuncOp"> {
  let summary = "Convert from StableHLO dialect to EmitC dialect.";
  let constructor = "createConvertStablehloToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertArithToEmitC : Pass<"convert-arith-to-emitc", "func::FuncOp"> {
  let summary = "Convert arith dialect to EmitC dialect, replacing IndexCastOp.";
  let constructor = "createConvertArithToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertTensorToEmitC : Pass<"convert-tensor-to-emitc", "func::FuncOp"> {
  let summary = "Convert tensor dialect to EmitC dialect, replacing ExtractOp.";
  let constructor = "createConvertTensorToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

def ConvertTosaToEmitC : Pass<"convert-tosa-to-emitc", "func::FuncOp"> {
  let summary = "Convert TOSA dialect to EmitC dialect.";
  let constructor = "createConvertTosaToEmitCPass()";
  let dependentDialects = ["EmitCDialect"];
}

#endif // EMITC_CONVERSION_PASSES
