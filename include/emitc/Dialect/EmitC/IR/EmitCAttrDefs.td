//===- EmitCAttrDefs.td - EmitC attributes -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the MLIR EmitC attributes.
//
//===----------------------------------------------------------------------===//


#ifndef EMITC_DIALECT_EMITC_IR_EMITCATTRDEFS
#define EMITC_DIALECT_EMITC_IR_EMITCATTRDEFS

include "emitc/Dialect/EmitC/IR/EmitCBase.td"

//===----------------------------------------------------------------------===//
// EmitC attribute definitions
//===----------------------------------------------------------------------===//

class EmitC_Attr<string name, string attrMnemonic> : AttrDef<EmitC_Dialect, name> {
  let mnemonic = attrMnemonic;
}

def EmitC_OpaqueAttr : EmitC_Attr<"Opaque", "opaque"> {
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
}

#endif // EMITC_DIALECT_EMITC_IR_EMITCATTRDEFS
