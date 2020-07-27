//===- EmitC.td - EmitC operation definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines some operations with trivial mapping to C/C++ constructs.
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_OPS
#define EMITC_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def EmitC_Dialect : Dialect {
  let name = "emitc";
  let cppNamespace = "emitc";
}

// Base class for EmitC dialect ops.
class EmitC_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<EmitC_Dialect, mnemonic, traits>;

def EmitC_CallOp : EmitC_Op<"call", []> {
  let summary = "call operation";
  let description = [{
    The "call" operation represents a C++ function call. The call allows
    specifying order of operands and attributes in the call as follows:

    - integer value of index type refers to an operand;
    - attribute which will get lowered to constant value in call;
  }];
  let arguments = (ins
    Arg<StrAttr, "the C++ function to call">:$callee,
    Arg<OptionalAttr<ArrayAttr>, "the order of operands and attributes">:$args,
    Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def EmitC_YieldOp : EmitC_Op<"yield", [NoSideEffect, ReturnLike, Terminator]> {
//                               ParentOneOf<["IfOp, ForOp"]>]> {
  let summary = "loop yield and termination operation";
  let description = [{
    "emitc.yield" yields an SSA value from the EmitC dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "emitc.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "emitc.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result",
              [{ /* nothing to do */ }]>
  ];
  // Override default verifier (defined in SCF_Op), no custom verification
  // needed.
  //let verifier = ?;
}

#endif // EMITC_OPS
