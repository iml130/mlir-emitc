//===- EmitCOps.cpp - MLIR EmitC ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the EmitC dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace emitc;

//===----------------------------------------------------------------------===//
// EmitCDialect
//===----------------------------------------------------------------------===//

void emitc::EmitCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "emitc/Dialect/EmitC/EmitC.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "emitc/Dialect/EmitC/EmitCTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "emitc/Dialect/EmitC/EmitCAttrDefs.cpp.inc"
      >();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *emitc::EmitCDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  return builder.create<ConstantOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(mlir::emitc::ApplyOp op) {
  StringRef applicableOperator = op.applicableOperator();

  // Applicable operator must not be empty.
  if (applicableOperator.empty()) {
    return op.emitOpError("applicable operator must not be empty");
  }

  // Only `*` and `&` are supported.
  if (!applicableOperator.equals("&") && !applicableOperator.equals("*"))
    return op.emitOpError("applicable operator is illegal");

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(mlir::emitc::CallOp op) {
  // Callee must not be empty.
  if (op.callee().empty()) {
    return op.emitOpError("callee must not be empty");
  }

  auto argsAttr = op.args();
  if (argsAttr.hasValue()) {
    for (auto &arg : argsAttr.getValue()) {
      if (auto iArg = arg.dyn_cast<IntegerAttr>()) {
        if (iArg.getType().isIndex()) {
          int64_t index = iArg.getInt();
          // Args with elements of type index must be in range
          // [0..operands.size).
          if ((index < 0) ||
              (index >= static_cast<int64_t>(op.getNumOperands()))) {
            return op.emitOpError("index argument is out of range");
          }
        }
      }
      // Args with elements of type ArrayAttr must have a type.
      else if (auto aArg = arg.dyn_cast<ArrayAttr>()) {
        if (aArg.getType().isa<NoneType>()) {
          return op.emitOpError("array argument has no type");
        }
      }
    }
  }

  auto templateArgsAttr = op.template_args();
  if (templateArgsAttr.hasValue()) {
    for (auto &tArg : templateArgsAttr.getValue()) {
      // C++ forbids float literals as template arguments.
      if (auto iArg = tArg.dyn_cast<FloatAttr>()) {
        return op.emitOpError("float literal as template argument is invalid");
      }
      // Template args with elements of type ArrayAttr are not allowed.
      else if (auto aArg = tArg.dyn_cast<ArrayAttr>()) {
        return op.emitOpError("array as template arguments is invalid");
      }
      // Template args with elements of type DenseElementsAttr are not
      // allowed.
      else if (auto dArg = tArg.dyn_cast<DenseElementsAttr>()) {
        return op.emitOpError("dense elements as template "
                              "argument are invalid");
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

/// The constant op requires that the attribute's type matches the return type.
static LogicalResult verify(ConstOp &op) {
  auto value = op.value();
  Type type = op.getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return op.emitOpError() << "requires attribute's type (" << value.getType()
                            << ") to match op's return type (" << type << ")";
  return success();
}

// Folder.
OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return value();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "emitc/Dialect/EmitC/EmitC.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "emitc/Dialect/EmitC/EmitCAttrDefs.cpp.inc"

Attribute emitc::EmitCDialect::parseAttribute(DialectAsmParser &parser,
                                              Type type) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Attribute();
  Attribute genAttr;
  auto parseResult =
      generatedAttributeParser(getContext(), parser, mnemonic, type, genAttr);
  if (parseResult.hasValue())
    return genAttr;
  parser.emitError(typeLoc, "unknown attribute in EmitC dialect");
  return Attribute();
}

void emitc::EmitCDialect::printAttribute(Attribute attr,
                                         DialectAsmPrinter &os) const {
  if (failed(generatedAttributePrinter(attr, os)))
    llvm_unreachable("unexpected 'EmitC' attribute kind");
}

#define GET_TYPEDEF_CLASSES
#include "emitc/Dialect/EmitC/EmitCTypes.cpp.inc"

Type emitc::EmitCDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();
  Type genType;
  auto parseResult =
      generatedTypeParser(getContext(), parser, mnemonic, genType);
  if (parseResult.hasValue())
    return genType;
  parser.emitError(typeLoc, "unknown type in EmitC dialect");
  return Type();
}

void emitc::EmitCDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (failed(generatedTypePrinter(type, os)))
    llvm_unreachable("unexpected 'EmitC' type kind");
}
