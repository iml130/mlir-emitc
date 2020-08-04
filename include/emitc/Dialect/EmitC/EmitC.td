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

def EmitC_IfOp : EmitC_Op<"if",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"YieldOp">, RecursiveSideEffects,
       NoRegionArguments]> {
  let summary = "if-then-else operation";
  let description = [{
    The `emitc.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    emitc.if %b  {
      ...
    } else {
      ...
    }
    ```

    `emitc.if` may also return results that are defined in its regions. The
    values defined are determined by which execution path is taken.

    Example:

    ```mlir
    %x, %y = emitc.if %b -> (f32, f32) {
      %x_true = ...
      %y_true = ...
      emitc.yield %x_true, %y_true : f32, f32
    } else {
      %x_false = ...
      %y_false = ...
      emitc.yield %x_false, %y_false : f32, f32
    }
    ```

    `emitc.if` regions are always terminated with "emitc.yield". If "emitc.if"
    defines no values, the "emitc.yield" can be left out, and will be inserted
    implicitly. Otherwise, it must be explicit.
    Also, if "emitc.if" defines one or more values, the 'else' block cannot be
    omitted.

    Example:

    ```mlir
    emitc.if %b  {
      ...
    }
    ```
  }];
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value cond, bool withElseRegion">,
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "TypeRange resultTypes, Value cond, bool withElseRegion">,
    OpBuilder<
        "OpBuilder &builder, OperationState &result, TypeRange resultTypes, "
        "Value cond, "
        "function_ref<void(OpBuilder &, Location)> thenBuilder "
        "    = buildTerminatedBody, "
        "function_ref<void(OpBuilder &, Location)> elseBuilder = nullptr">,
    OpBuilder<
        "OpBuilder &builder, OperationState &result, Value cond, "
        "function_ref<void(OpBuilder &, Location)> thenBuilder "
        "    = buildTerminatedBody, "
        "function_ref<void(OpBuilder &, Location)> elseBuilder = nullptr">
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder() {
      Block* body = getBody(0);
      return results().empty() ? OpBuilder::atBlockTerminator(body)
                               : OpBuilder::atBlockEnd(body);
    }
    OpBuilder getElseBodyBuilder() {
      Block* body = getBody(1);
      return results().empty() ? OpBuilder::atBlockTerminator(body)
                               : OpBuilder::atBlockEnd(body);
    }
  }];

  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
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
