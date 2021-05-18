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

//===----------------------------------------------------------------------===//
// EmitC dialect definition
//===----------------------------------------------------------------------===//

def EmitC_Dialect : Dialect {
  let name = "emitc";
  let cppNamespace = "::mlir::emitc";
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// EmitC attribute definitions
//===----------------------------------------------------------------------===//

class EmitC_Attr<string name> : AttrDef<EmitC_Dialect, name> { }

def OpaqueAttr : EmitC_Attr<"Opaque"> {
  let mnemonic = "opaque";

  let summary = "An opaque attribute";

  let description = [{
    An opaque attribute of which the value gets emitted as is.

    Examples:

    ```mlir
    #emitc.opaque<"">
    #emitc.opaque<"NULL">
    #emitc.opaque<"nullptr">
    ```
  }];

  let parameters = (ins StringRefParameter<"the opaque value">:$value);

  let printer = [{
    $_printer << "opaque<\"" << getValue() << "\">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Attribute();
    StringRef value;
    auto loc = $_parser.getCurrentLocation();
    if ($_parser.parseOptionalString(&value)) {
      $_parser.emitError(loc) << "expected string";
      return Attribute();
    }
    if ($_parser.parseGreater())
      return Attribute();
    return get($_ctxt, value);
  }];
}

//===----------------------------------------------------------------------===//
// EmitC type definitions
//===----------------------------------------------------------------------===//

class EmitC_Type<string name> : TypeDef<EmitC_Dialect, name> { }

def OpaqueType : EmitC_Type<"Opaque"> {
  let mnemonic = "opaque";

  let summary = "An opaque type";

  let description = [{
    An opaque data type of which the value gets emitted as is.

    Examples:

    ```mlir
    !emitc.opaque<"int">
    !emitc.opaque<"float *">
    !emitc.opaque<"std::vector<std::string>">
    ```
  }];

  let parameters = (ins StringRefParameter<"the opaque value">:$value);

  let printer = [{
    $_printer << "opaque<\"" << getValue() << "\">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    StringRef value;
    auto loc = $_parser.getCurrentLocation();
    if ($_parser.parseOptionalString(&value) || value.empty()) {
      $_parser.emitError(loc) << "expected non empty string";
      return Type();
    }
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, value);
  }];
}

//===----------------------------------------------------------------------===//
// EmitC op definitions
//===----------------------------------------------------------------------===//

// Base class for EmitC dialect ops.
class EmitC_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<EmitC_Dialect, mnemonic, traits>;

def EmitC_ApplyOp : EmitC_Op<"apply", []> {
  let summary = "apply operation";
  let description = [{
    With the "apply" operation the operators & (address of) and * (contents of)
    can be applied to a single operand.
  }];
  let arguments = (ins
    Arg<StrAttr, "the operator to apply">:$applicableOperator,
    AnyType:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $applicableOperator `(` $operand `)` attr-dict `:` functional-type($operand, results)
  }];
  let verifier = [{ return ::verify(*this); }];
}

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
    Arg<OptionalAttr<ArrayAttr>, "template arguments">:$template_args,
    Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let verifier = [{ return ::verify(*this); }];
}

def EmitC_ConstOp : EmitC_Op<"const", [ConstantLike, NoSideEffect]> {
  let summary = "Constant op";
  let description = [{
    The `const` operation produces an SSA value equal to some constant
    specified by an attribute, similar to `std.constant`.
    In addition to the `std.constant` operation, the `emitc.const` operation
    is intended to support opaque attributes and EmitC's opaque type.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let verifier = [{ return ::verify(*this); }];

  let hasFolder = 1;
}

def EmitC_IncludeOp : EmitC_Op<"include", [NoSideEffect, HasParent<"ModuleOp">]> {
  let summary = "include operation";
  let description = [{
    The `include` operation allows to define a source file inclusion via the
    `#include` directive.
  }];
  let arguments = (ins
    Arg<StrAttr, "source file to include">:$include,
    UnitAttr:$is_standard_include);
  let assemblyFormat = [{
    $include attr-dict (`is_standard_include` $is_standard_include^)?
  }];
}

#endif // EMITC_OPS
