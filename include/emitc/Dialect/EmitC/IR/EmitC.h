//===- EmitC.h - EmitC Dialect ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares EmitC in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_DIALECT_EMITC_IR_EMITC_H
#define EMITC_DIALECT_EMITC_IR_EMITC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "emitc/Dialect/EmitC/IR/EmitCDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "emitc/Dialect/EmitC/IR/EmitCAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "emitc/Dialect/EmitC/IR/EmitCTypes.h.inc"

#define GET_OP_CLASSES
#include "emitc/Dialect/EmitC/IR/EmitC.h.inc"

#endif // EMITC_DIALECT_EMITC_IR_EMITC_H
