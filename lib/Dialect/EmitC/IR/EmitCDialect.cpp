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
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace emitc;

//===----------------------------------------------------------------------===//
// EmitCDialect
//===----------------------------------------------------------------------===//

emitc::EmitCDialect::EmitCDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "emitc/Dialect/EmitC/EmitC.cpp.inc"
      >();
  allowUnknownTypes();
  allowUnknownOperations();
}

/// Default callback for IfOp builders. Inserts a yield without arguments.
void emitc::buildTerminatedBody(OpBuilder &builder, Location loc) {
  builder.create<emitc::YieldOp>(loc);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//
// Adopted from SCF dialect

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 bool withElseRegion) {
  build(builder, result, /*resultTypes=*/llvm::None, cond, withElseRegion);
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value cond, bool withElseRegion) {
  auto addTerminator = [&](OpBuilder &nested, Location loc) {
    if (resultTypes.empty())
      IfOp::ensureTerminator(*nested.getInsertionBlock()->getParent(), nested,
                             loc);
  };

  build(builder, result, resultTypes, cond, addTerminator,
        withElseRegion ? addTerminator
                       : function_ref<void(OpBuilder &, Location)>());
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value cond,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  assert(thenBuilder && "the builder callback for 'then' must be present");

  result.addOperands(cond);
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  Region *elseRegion = result.addRegion();
  if (!elseBuilder)
    return;

  builder.createBlock(elseRegion);
  elseBuilder(builder, result.location);
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  build(builder, result, TypeRange(), cond, thenBuilder, elseBuilder);
}

static LogicalResult verify(IfOp op) {
  if (op.getNumResults() != 0 && op.elseRegion().empty())
    return op.emitOpError("must have an else block if defining values");

  return RegionBranchOpInterface::verifyTypes(op);
}

static ParseResult parseIfOp(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
  Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();
  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    IfOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, IfOp op) {
  bool printBlockTerminators = false;

  p << IfOp::getOperationName() << " " << op.condition();
  if (!op.results().empty()) {
    p << " -> (" << op.getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p.printRegion(op.thenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    p << " else";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printBlockTerminators);
  }

  p.printOptionalAttrDict(op.getAttrs());
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void IfOp::getSuccessorRegions(Optional<unsigned> index,
                               ArrayRef<Attribute> operands,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->elseRegion();
  if (elseRegion->empty())
    elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  bool condition;
  if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
    condition = condAttr.getValue().isOneValue();
  } else {
    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&thenRegion()));
    regions.push_back(RegionSuccessor(elseRegion));
    return;
  }

  // Add the successor regions using the condition.
  regions.push_back(RegionSuccessor(condition ? &thenRegion() : elseRegion));
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace emitc {
#define GET_OP_CLASSES
#include "emitc/Dialect/EmitC/EmitC.cpp.inc"
} // namespace emitc
} // namespace mlir
