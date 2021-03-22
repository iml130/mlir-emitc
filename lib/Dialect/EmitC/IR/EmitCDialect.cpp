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
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *emitc::EmitCDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  return builder.create<ConstantOp>(loc, type, value);
}

/// Default callback for IfOp builders. Inserts a yield without arguments.
void emitc::buildTerminatedBody(OpBuilder &builder, Location loc) {
  builder.create<emitc::YieldOp>(loc);
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
// Folder

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return value();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//
// Adopted from SCF dialect.

void ForOp::build(OpBuilder &builder, OperationState &result, Value lb,
                  Value ub, Value step, ValueRange iterArgs,
                  BodyBuilderFn bodyBuilder) {
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);
  for (Value v : iterArgs)
    result.addTypes(v.getType());
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(builder.getIndexType());
  for (Value v : iterArgs)
    bodyBlock.addArgument(v.getType());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (iterArgs.empty() && !bodyBuilder) {
    ForOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArgument(0),
                bodyBlock.getArguments().drop_front());
  }
}

static LogicalResult verify(ForOp op) {
  if (auto cst = op.step().getDefiningOp<ConstantIndexOp>())
    if (cst.getValue() <= 0)
      return op.emitOpError("constant step operand must be positive");

  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (!body->getArgument(0).getType().isIndex())
    return op.emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = op.getNumResults();
  if (opNumResults == 0)
    return success();
  // If ForOp defines values, check that the number and types of
  // the defined values match ForOp initial iter operands and backedge
  // basic block arguments.
  if (op.getNumIterOperands() != opNumResults)
    return op.emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (op.getNumRegionIterArgs() != opNumResults)
    return op.emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = op.getIterOperands();
  auto iterArgs = op.getRegionIterArgs();
  auto opResults = op.getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter region arg and defined value";

    i++;
  }

  return RegionBranchOpInterface::verifyTypes(op);
}

static void print(OpAsmPrinter &p, ForOp op) {
  bool printBlockTerminators = false;
  p << op.getOperationName() << " " << op.getInductionVar() << " = "
    << op.lowerBound() << " to " << op.upperBound() << " step " << op.step();

  if (op.hasIterOperands()) {
    p << " iter_args(";
    auto regionArgs = op.getRegionIterArgs();
    auto operands = op.getIterOperands();

    llvm::interleaveComma(llvm::zip(regionArgs, operands), p, [&](auto it) {
      p << std::get<0>(it) << " = " << std::get<1>(it);
    });
    p << ")";
    p << " -> (" << op.getResultTypes() << ")";
    printBlockTerminators = true;
  }
  p.printRegion(op.region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);
  p.printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseForOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return failure();

  // Parse loop bounds.
  Type indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::OperandType, 4> regionArgs, operands;
  SmallVector<Type, 4> argTypes;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    // Resolve input operands.
    for (auto operand_type : llvm::zip(operands, result.types))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return failure();
  }
  // Induction variable.
  argTypes.push_back(indexType);
  // Loop carried variables
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  Region *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs, argTypes))
    return failure();

  ForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

Region &ForOp::getLoopBody() { return region(); }

bool ForOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult ForOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto op : ops)
    op->moveBefore(*this);
  return success();
}

ForOp mlir::emitc::getForInductionVarOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return ForOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<ForOp>(containingOp);
}

/// Return operands used when entering the region at 'index'. These operands
/// correspond to the loop iterator operands, i.e., those exclusing the
/// induction variable. LoopOp only has one region, so 0 is the only valid value
/// for `index`.
OperandRange ForOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0 && "invalid region index");

  // The initial operands map to the loop arguments after the induction
  // variable.
  return initArgs();
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ForOp::getSuccessorRegions(Optional<unsigned> index,
                                ArrayRef<Attribute> operands,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the ForOp, branch into the body using the iterator
  // arguments.
  if (!index.hasValue()) {
    regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
    return;
  }

  // Otherwise, the loop may branch back to itself or the parent operation.
  assert(index.getValue() == 0 && "expected loop region");
  regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
  regions.push_back(RegionSuccessor(getResults()));
}

ValueVector mlir::emitc::buildLoopNest(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps, ValueRange iterArgs,
    function_ref<ValueVector(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilder) {
  assert(lbs.size() == ubs.size() &&
         "expected the same number of lower and upper bounds");
  assert(lbs.size() == steps.size() &&
         "expected the same number of lower bounds and steps");

  // If there are no bounds, call the body-building function and return early.
  if (lbs.empty()) {
    ValueVector results =
        bodyBuilder ? bodyBuilder(builder, loc, ValueRange(), iterArgs)
                    : ValueVector();
    assert(results.size() == iterArgs.size() &&
           "loop nest body must return as many values as loop has iteration "
           "arguments");
    return results;
  }

  // First, create the loop structure iteratively using the body-builder
  // callback of `ForOp::build`. Do not create `YieldOp`s yet.
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<emitc::ForOp, 4> loops;
  SmallVector<Value, 4> ivs;
  loops.reserve(lbs.size());
  ivs.reserve(lbs.size());
  ValueRange currentIterArgs = iterArgs;
  Location currentLoc = loc;
  for (unsigned i = 0, e = lbs.size(); i < e; ++i) {
    auto loop = builder.create<emitc::ForOp>(
        currentLoc, lbs[i], ubs[i], steps[i], currentIterArgs,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange args) {
          ivs.push_back(iv);
          // It is safe to store ValueRange args because it points to block
          // arguments of a loop operation that we also own.
          currentIterArgs = args;
          currentLoc = nestedLoc;
        });
    // Set the builder to point to the body of the newly created loop. We don't
    // do this in the callback because the builder is reset when the callback
    // returns.
    builder.setInsertionPointToStart(loop.getBody());
    loops.push_back(loop);
  }

  // For all loops but the innermost, yield the results of the nested loop.
  for (unsigned i = 0, e = loops.size() - 1; i < e; ++i) {
    builder.setInsertionPointToEnd(loops[i].getBody());
    builder.create<emitc::YieldOp>(loc, loops[i + 1].getResults());
  }

  // In the body of the innermost loop, call the body building function if any
  // and yield its results.
  builder.setInsertionPointToStart(loops.back().getBody());
  ValueVector results = bodyBuilder
                            ? bodyBuilder(builder, currentLoc, ivs,
                                          loops.back().getRegionIterArgs())
                            : ValueVector();
  assert(results.size() == iterArgs.size() &&
         "loop nest body must return as many values as loop has iteration "
         "arguments");
  builder.setInsertionPointToEnd(loops.back().getBody());
  builder.create<emitc::YieldOp>(loc, results);

  // Return the results of the outermost loop.
  return ValueVector(loops.front().result_begin(), loops.front().result_end());
}

ValueVector mlir::emitc::buildLoopNest(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  // Delegate to the main function by wrapping the body builder.
  return buildLoopNest(builder, loc, lbs, ubs, steps, llvm::None,
                       [&bodyBuilder](OpBuilder &nestedBuilder,
                                      Location nestedLoc, ValueRange ivs,
                                      ValueRange) -> ValueVector {
                         if (bodyBuilder)
                           bodyBuilder(nestedBuilder, nestedLoc, ivs);
                         return {};
                       });
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//
// Adopted from SCF dialect.

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

  p.printOptionalAttrDict(op->getAttrs());
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
// YieldOp
//===----------------------------------------------------------------------===//
// Adopted from SCF dialect.

static ParseResult parseYieldOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse variadic operands list, their types, and resolve operands to SSA
  // values.
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalColonTypeList(types) ||
      parser.resolveOperands(operands, types, loc, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, emitc::YieldOp op) {
  p << op.getOperationName();
  if (op.getNumOperands() != 0)
    p << ' ' << op.getOperands() << " : " << op.getOperandTypes();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "emitc/Dialect/EmitC/EmitC.cpp.inc"

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
