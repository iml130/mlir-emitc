//===- Passes.td - EmitC pass definition file --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_DIALECT_EMITC_TRANSFORMS_PASSES
#define EMITC_DIALECT_EMITC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def InsertEmitCMHLOInclude : Pass<"insert-emitc-mhlo-include", "ModuleOp"> {
  let summary = "Insert an EmitC include for the MHLO dialect.";
  let constructor = "createInsertEmitCMHLOIncludePass()";
  let dependentDialects = ["EmitCDialect"];
}

def InsertEmitCArithInclude : Pass<"insert-emitc-arith-include", "ModuleOp"> {
  let summary = "Insert an EmitC include for the arith dialect.";
  let constructor = "createInsertEmitCArithIncludePass()";
  let dependentDialects = ["EmitCDialect"];
}

def InsertEmitCStdInclude : Pass<"insert-emitc-std-include", "ModuleOp"> {
  let summary = "Insert an EmitC include for the std dialect.";
  let constructor = "createInsertEmitCStdIncludePass()";
  let dependentDialects = ["EmitCDialect"];
}

def InsertEmitCTensorInclude : Pass<"insert-emitc-tensor-include", "ModuleOp"> {
  let summary = "Insert an EmitC include for the tensor dialect.";
  let constructor = "createInsertEmitCTensorIncludePass()";
  let dependentDialects = ["EmitCDialect"];
}

def InsertEmitCTosaInclude : Pass<"insert-emitc-tosa-include", "ModuleOp"> {
  let summary = "Insert an EmitC include for the TOSA dialect.";
  let constructor = "createInsertEmitCTosaIncludePass()";
  let dependentDialects = ["EmitCDialect"];
}

#endif // EMITC_DIALECT_EMITC_TRANSFORMS_PASSES
