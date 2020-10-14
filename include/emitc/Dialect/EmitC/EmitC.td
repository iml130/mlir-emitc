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
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def EmitC_Dialect : Dialect {
  let name = "emitc";
  let cppNamespace = "::mlir::emitc";
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
    Arg<OptionalAttr<ArrayAttr>, "the order of operands and attributes">:$template_args,
    Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def EmitC_ForOp : EmitC_Op<"for",
      [DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"emitc::YieldOp">,
       RecursiveSideEffects]> {
  let summary = "for operation";
  let description = [{
    The "emitc.for" operation represents a loop taking 3 SSA value as operands
    that represent the lower bound, upper bound and step respectively.  The
    operation defines an SSA value for its induction variable. It has one
    region capturing the loop body. The induction variable is represented as an
    argument of this region. This SSA value always has type index, which is the
    size of the machine word. The step is a value of type index, required to be
    positive.
    The lower and upper bounds specify a half-open range: the range includes
    the lower bound but does not include the upper bound.

    The body region must contain exactly one block that terminates with
    "emitc.yield". Calling ForOp::build will create such a region and insert
    the terminator implicitly if none is defined, so will the parsing even in
    cases when it is absent from the custom format. For example:

    ```mlir
    emitc.for %iv = %lb to %ub step %step {
      ... // body
    }
    ```

    `emitc.for` can also operate on loop-carried variables and returns the final
    values after loop termination. The initial values of the variables are
    passed as additional SSA operands to the "emitc.for" following the 3 loop
    control SSA values mentioned above (lower bound, upper bound and step). The
    operation region has equivalent arguments for each variable representing
    the value of the variable at the current iteration.

    The region must terminate with a "emitc.yield" that passes all the current
    iteration variables to the next iteration, or to the "emitc.for" result, if
    at the last iteration. Note, that when the loop-carried variables are
    present, calling ForOp::build will not insert the terminator implicitly.
    The caller must insert "emitc.yield" in that case.

    "emitc.for" results hold the final values after the last iteration.
    For example, to sum-reduce a memref:

    ```mlir
    func @reduce(%buffer: memref<1024xf32>, %lb: index,
                 %ub: index, %step: index) -> (f32) {
      // Initial sum set to 0.
      %sum_0 = constant 0.0 : f32
      // iter_args binds initial values to the loop's region arguments.
      %sum = emitc.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = load %buffer[%iv] : memref<1024xf32>
        %sum_next = addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum
        // if final iteration.
        emitc.yield %sum_next : f32
      }
      return %sum : f32
    }
    ```

    If the "emitc.for" defines any values, a yield must be explicitly present.
    The number and types of the "emitc.for" results must match the initial
    values in the "iter_args" binding and the yield operands.

    Another example with a nested "emitc.if" (see "emitc.if" for details) to
    perform conditional reduction:

    ```mlir
    func @conditional_reduce(%buffer: memref<1024xf32>, %lb: index,
                             %ub: index, %step: index) -> (f32) {
      %sum_0 = constant 0.0 : f32
      %c0 = constant 0.0 : f32
      %sum = emitc.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = load %buffer[%iv] : memref<1024xf32>
        %cond = cmpf "ugt", %t, %c0 : f32
        %sum_next = emitc.if %cond -> (f32) {
          %new_sum = addf %sum_iter, %t : f32
          emitc.yield %new_sum : f32
        } else {
          emitc.yield %sum_iter : f32
        }
        emitc.yield %sum_next : f32
      }
      return %sum : f32
    }
    ```
  }];
  let arguments = (ins Index:$lowerBound,
                       Index:$upperBound,
                       Index:$step,
                       Variadic<AnyType>:$initArgs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Value lowerBound, Value upperBound, Value step, "
              "ValueRange iterArgs = llvm::None, "
              "function_ref<void(OpBuilder &, Location, Value, ValueRange)>"
              "    = nullptr">
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    Value getInductionVar() { return getBody()->getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front();
    }
    Operation::operand_range getIterOperands() {
      return getOperands().drop_front(getNumControlOperands());
    }

    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }

    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - 1;
    }
    /// Number of operands controlling the loop: lb, ub, step
    unsigned getNumControlOperands() { return 3; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() {
      return getOperation()->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return getOperation()->getNumOperands() - getNumControlOperands();
    }

    /// Return operands used when entering the region at 'index'. These operands
    /// correspond to the loop iterator operands, i.e., those exclusing the
    /// induction variable. LoopOp only has one region, so 0 is the only valid
    /// value for `index`.
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];

  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def EmitC_IfOp : EmitC_Op<"if",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"emitc::YieldOp">, RecursiveSideEffects,
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
    OpBuilder<"Value cond, bool withElseRegion">,
    OpBuilder<"TypeRange resultTypes, Value cond, bool withElseRegion">,
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
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

#endif // EMITC_OPS
