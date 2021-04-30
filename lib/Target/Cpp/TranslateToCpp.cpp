//===- TranslateToCpp.cpp - Translating to C++ calls ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "emitc/Target/Cpp.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "translate-to-cpp"

using namespace mlir;
using namespace mlir::emitc;
using llvm::formatv;

template <typename ConstOpTy>
static LogicalResult printConstantOp(CppEmitter &emitter,
                                     ConstOpTy constantOp) {
  auto &os = emitter.ostream();
  auto result = constantOp.getOperation()->getResult(0);
  auto value = constantOp.value();

  bool isScalar = !result.getType().template isa<TensorType>();

  // Add braces only if
  //  - cpp code is emitted,
  //  - variables are not forward declared
  //  - and the emitted type is a scalar (to prevent double brace
  //  initialization).
  bool braceInitialization =
      !emitter.restrictedToC() && !emitter.forwardDeclaredVariables();
  bool emitBraces = braceInitialization && isScalar;

  // Emit an assignment if variables are forward declared.
  if (emitter.forwardDeclaredVariables()) {
    // Special case for emitc.const.
    if (auto sAttr = value.template dyn_cast<StringAttr>()) {
      if (sAttr.getValue().empty())
        return success();
    }

    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    if (failed(emitter.emitAttribute(*constantOp.getOperation(), value)))
      return failure();
    return success();
  }

  // Special case for emitc.const.
  if (auto sAttr = value.template dyn_cast<StringAttr>()) {
    if (sAttr.getValue().empty()) {
      // The semicolon gets printed by the emitOperation function.
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/false)))
        return failure();
      return success();
    }
  }

  // We have to emit a variable declaration.
  if (!braceInitialization) {
    // If brace initialization is not used, we have to emit an assignment.
    if (failed(emitter.emitAssignPrefix(*constantOp.getOperation()))) {
      return failure();
    }
    if (failed(emitter.emitAttribute(*constantOp.getOperation(), value)))
      return failure();
    return success();
  }

  if (failed(emitter.emitVariableDeclaration(result,
                                             /*trailingSemicolon=*/false)))
    return failure();

  if (emitBraces)
    os << "{";
  if (failed(emitter.emitAttribute(*constantOp.getOperation(), value)))
    return failure();
  if (emitBraces)
    os << "}";

  return success();
}

static LogicalResult printBranchOp(CppEmitter &emitter,
                                   mlir::BranchOp branchOp) {
  auto &os = emitter.ostream();
  Block &successor = *branchOp.getSuccessor();

  for (auto pair :
       llvm::zip(branchOp.getOperands(), successor.getArguments())) {
    auto &operand = std::get<0>(pair);
    auto &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(successor))) {
    return branchOp.emitOpError() << "Unable to find label for successor block";
  }
  os << emitter.getOrCreateName(successor);
  return success();
}

static LogicalResult printCondBranchOp(CppEmitter &emitter,
                                       mlir::CondBranchOp condBranchOp) {
  auto &os = emitter.ostream();
  Block &trueSuccessor = *condBranchOp.getTrueDest();
  Block &falseSuccessor = *condBranchOp.getFalseDest();

  os << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
     << ") {\n";

  // If condition is true.
  for (auto pair : llvm::zip(condBranchOp.getTrueOperands(),
                             trueSuccessor.getArguments())) {
    auto &operand = std::get<0>(pair);
    auto &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(trueSuccessor))) {
    return condBranchOp.emitOpError()
           << "Unable to find label for successor block";
  }
  os << emitter.getOrCreateName(trueSuccessor) << ";\n";
  os << "} else {\n";
  // If condition is false.
  for (auto pair : llvm::zip(condBranchOp.getFalseOperands(),
                             falseSuccessor.getArguments())) {
    auto &operand = std::get<0>(pair);
    auto &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(falseSuccessor))) {
    return condBranchOp.emitOpError()
           << "Unable to find label for successor block";
  }
  os << emitter.getOrCreateName(falseSuccessor) << ";\n";
  os << "}";
  return success();
}

static LogicalResult printCallOp(CppEmitter &emitter, mlir::CallOp callOp) {
  if (failed(emitter.emitAssignPrefix(*callOp.getOperation())))
    return failure();

  auto &os = emitter.ostream();
  os << callOp.getCallee() << "(";
  if (failed(emitter.emitOperands(*callOp.getOperation())))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printCallOp(CppEmitter &emitter, emitc::CallOp callOp) {
  auto &os = emitter.ostream();
  auto &op = *callOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << callOp.callee();

  auto emitArgs = [&](Attribute attr) -> LogicalResult {
    if (auto t = attr.dyn_cast<IntegerAttr>()) {
      // Index attributes are treated specially as operand index.
      if (t.getType().isIndex()) {
        auto idx = t.getInt();
        if ((idx < 0) || (idx >= op.getNumOperands()))
          return op.emitOpError() << "invalid operand index";
        if (!emitter.hasValueInScope(op.getOperand(idx)))
          return op.emitOpError()
                 << "operand " << idx << "'s value not defined in scope";
        os << emitter.getOrCreateName(op.getOperand(idx));
        return success();
      }
    }
    if (failed(emitter.emitAttribute(op, attr))) {
      return failure();
    }
    return success();
  };

  if (callOp.template_args()) {
    if (emitter.restrictedToC()) {
      return op.emitOpError()
             << "template arguments are not supported if emitting C";
    }
    os << "<";
    if (failed(interleaveCommaWithError(*callOp.template_args(), os, emitArgs)))
      return failure();
    os << ">";
  }

  os << "(";

  auto emittedArgs =
      callOp.args() ? interleaveCommaWithError(*callOp.args(), os, emitArgs)
                    : emitter.emitOperands(op);
  if (failed(emittedArgs))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printApplyOp(CppEmitter &emitter, emitc::ApplyOp applyOp) {
  auto &os = emitter.ostream();
  auto &op = *applyOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << applyOp.applicableOperator();
  os << emitter.getOrCreateName(applyOp.getOperand());

  return success();
}

static LogicalResult printForOp(CppEmitter &emitter, emitc::ForOp forOp) {

  auto &os = emitter.ostream();

  auto operands = forOp.getIterOperands();
  auto iterArgs = forOp.getRegionIterArgs();
  auto results = forOp.getResults();

  if (!emitter.forwardDeclaredVariables()) {
    for (auto result : forOp.getResults()) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  for (auto pair : llvm::zip(iterArgs, operands)) {
    if (failed(emitter.emitType(*forOp.getOperation(),
                                std::get<0>(pair).getType())))
      return failure();
    os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    os << emitter.getOrCreateName(std::get<1>(pair)) << ";";
    os << "\n";
  }

  os << "for (";
  if (failed(emitter.emitType(*forOp.getOperation(),
                              forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());

  if (emitter.restrictedToC()) {
    os << " = ";
    os << emitter.getOrCreateName(forOp.lowerBound());
  } else {
    os << "{";
    os << emitter.getOrCreateName(forOp.lowerBound());
    os << "}";
  }

  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  os << emitter.getOrCreateName(forOp.upperBound());
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  os << emitter.getOrCreateName(forOp.step());
  os << ") {\n";

  auto &forRegion = forOp.region();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op because this updates the result variables
  // of the for op in the generated code. Instead we update the iterArgs at
  // the end of a loop iteration and set the result variables after the for
  // loop.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  auto yieldOp = forRegion.getBlocks().front().getTerminator();
  // Copy yield operands into iterArgs at the end of a loop iteration.
  for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
    auto iterArg = std::get<0>(pair);
    auto operand = std::get<1>(pair);
    os << emitter.getOrCreateName(iterArg) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "}\n";

  // Copy iterArgs into results after the for loop.
  for (auto pair : llvm::zip(results, iterArgs)) {
    auto result = std::get<0>(pair);
    auto iterArg = std::get<1>(pair);
    os << emitter.getOrCreateName(result) << " = "
       << emitter.getOrCreateName(iterArg) << ";\n";
  }

  return success();
}

static LogicalResult printIfOp(CppEmitter &emitter, emitc::IfOp ifOp) {
  auto &os = emitter.ostream();

  if (!emitter.forwardDeclaredVariables()) {
    for (auto result : ifOp.getResults()) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  os << "if (";
  if (failed(emitter.emitOperands(*ifOp.getOperation())))
    return failure();
  os << ") {\n";

  auto &thenRegion = ifOp.thenRegion();
  for (auto &op : thenRegion.getOps()) {
    // Note: This prints a superfluous semicolon if the terminating yield op has
    // zero results.
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
      return failure();
  }

  os << "}\n";

  auto &elseRegion = ifOp.elseRegion();
  if (!elseRegion.empty()) {
    os << "else {\n";

    for (auto &op : elseRegion.getOps()) {
      // Note: This prints a superfluous semicolon if the terminating yield op
      // has zero results.
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
        return failure();
    }

    os << "}\n";
  }

  return success();
}

static LogicalResult printYieldOp(CppEmitter &emitter, emitc::YieldOp yieldOp) {
  auto &os = emitter.ostream();
  auto &parentOp = *yieldOp.getOperation()->getParentOp();

  if (yieldOp.getNumOperands() != parentOp.getNumResults()) {
    return yieldOp.emitError("number of operands does not to match the number "
                             "of the parent op's results");
  }

  if (failed(interleaveWithError(
          llvm::zip(parentOp.getResults(), yieldOp.getOperands()),
          [&](auto pair) -> LogicalResult {
            auto result = std::get<0>(pair);
            auto operand = std::get<1>(pair);
            os << emitter.getOrCreateName(result) << " = ";

            if (!emitter.hasValueInScope(operand))
              return yieldOp.emitError() << "operand value not in scope";
            os << emitter.getOrCreateName(operand);
            return success();
          },
          [&]() { os << ";\n"; })))
    return failure();

  return success();
}

static LogicalResult printReturnOp(CppEmitter &emitter, ReturnOp returnOp) {
  auto &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " " << emitter.getOrCreateName(returnOp.getOperand(0));
    return success(emitter.hasValueInScope(returnOp.getOperand(0)));
  default:
    os << " std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }
}

static LogicalResult printModule(CppEmitter &emitter, ModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();

  if (emitter.restrictedToC()) {
    os << "#include <stdbool.h>\n";
    os << "#include <stddef.h>\n";
    os << "#include <stdint.h>\n\n";
  } else {
    os << "#include <cmath>\n\n";
    os << "#include \"emitc_mhlo.h\"\n";
    os << "#include \"emitc_std.h\"\n";
    os << "#include \"emitc_tensor.h\"\n";
    os << "#include \"emitc_tosa.h\"\n\n";
  }

  os << "// Forward declare functions.\n";
  for (FuncOp funcOp : moduleOp.getOps<FuncOp>()) {
    if (failed(emitter.emitTypes(*funcOp.getOperation(),
                                 funcOp.getType().getResults())))
      return failure();
    os << " " << funcOp.getName() << "(";
    if (failed(interleaveCommaWithError(
            funcOp.getArguments(), os, [&](BlockArgument arg) {
              return emitter.emitType(*funcOp.getOperation(), arg.getType());
            })))
      return failure();
    os << ");\n";
  }
  os << "\n";

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

static LogicalResult printFunction(CppEmitter &emitter, FuncOp functionOp) {
  // We need to forward-declare variables if the function has multiple blocks.
  if (!emitter.forwardDeclaredVariables() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError()
           << "with multiple blocks needs forward declared variables";
  }

  CppEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();
  if (failed(emitter.emitTypes(*functionOp.getOperation(),
                               functionOp.getType().getResults())))
    return failure();
  os << " " << functionOp.getName();

  os << "(";
  if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
            if (failed(emitter.emitType(*functionOp.getOperation(),
                                        arg.getType())))
              return failure();
            os << " " << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ") {\n";

  if (emitter.forwardDeclaredVariables()) {
    // We forward declare all result variables including results from ops
    // inside of regions.
    auto result =
        functionOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          for (auto result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(
                    result, /*trailingSemicolon=*/true))) {
              return WalkResult(
                  op->emitError("Unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();
  }

  auto &blocks = functionOp.getBlocks();
  // Create label names for basic blocks.
  for (auto &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Emit variables for basic block arguments.
  for (auto it = std::next(blocks.begin()); it != blocks.end(); ++it) {
    Block &block = *it;
    for (auto &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp.emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (failed(emitter.emitType(*block.getParentOp(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (auto &block : blocks) {
    // Only print a label if there is more than one block.
    if (blocks.size() > 1) {
      if (failed(emitter.emitLabel(block))) {
        return failure();
      }
    }
    for (Operation &op : block.getOperations()) {
      // Don't print additional semicolons after these operations.
      bool trailingSemicolon =
          !isa<emitc::IfOp, emitc::ForOp, mlir::CondBranchOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }
  os << "}\n";
  return success();
}

CppEmitter::CppEmitter(raw_ostream &os, bool restrictToC,
                       bool forwardDeclareVariables)
    : os(os), restrictToC(restrictToC),
      forwardDeclareVariables(forwardDeclareVariables) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

/// Return the existing or a new name for a Value.
StringRef CppEmitter::getOrCreateName(Value val) {
  if (!valMapper.count(val))
    valMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *valMapper.begin(val);
}

/// Return the existing or a new name for a Block.
StringRef CppEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

bool CppEmitter::mapToSigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return true;
  case IntegerType::Signed:
    return true;
  case IntegerType::Unsigned:
    return false;
  }
}

bool CppEmitter::hasValueInScope(Value val) { return valMapper.count(val); }

bool CppEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}

LogicalResult CppEmitter::emitAttribute(Operation &op, Attribute attr) {
  auto printInt = [&](APInt val, bool isSigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      val.print(os, isSigned);
    }
  };

  auto printFloat = [&](APFloat val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "(float)";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        os << "(double)";
        break;
      default:
        break;
      };
      os << strValue;
    } else if (val.isNaN()) {
      os << "NAN";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "-";
      os << "INFINITY";
    }
  };

  // Print floating point attributes.
  if (auto fAttr = attr.dyn_cast<FloatAttr>()) {
    printFloat(fAttr.getValue());
    return success();
  }
  auto denseErrorMessage = "dense attributes are not supported if emitting C";
  if (auto dense = attr.dyn_cast<mlir::DenseFPElementsAttr>()) {
    // Dense attributes are not supported if emitting C.
    if (restrictedToC())
      return op.emitError(denseErrorMessage);
    os << '{';
    interleaveComma(dense, os, [&](APFloat val) { printFloat(val); });
    os << '}';
    return success();
  }

  // Print int attributes.
  if (auto iAttr = attr.dyn_cast<IntegerAttr>()) {
    if (auto iType = iAttr.getType().dyn_cast<IntegerType>()) {
      printInt(iAttr.getValue(), mapToSigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = iAttr.getType().dyn_cast<IndexType>()) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = attr.dyn_cast<DenseIntElementsAttr>()) {
    // Dense attributes are not supported if emitting C.
    if (restrictedToC())
      return op.emitError(denseErrorMessage);
    if (auto iType = dense.getType()
                         .cast<TensorType>()
                         .getElementType()
                         .dyn_cast<IntegerType>()) {
      os << '{';
      interleaveComma(dense, os, [&](APInt val) {
        printInt(val, mapToSigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dense.getType()
                         .cast<TensorType>()
                         .getElementType()
                         .dyn_cast<IndexType>()) {
      os << '{';
      interleaveComma(dense, os, [&](APInt val) { printInt(val, false); });
      os << '}';
      return success();
    }
  }
  if (auto sAttr = attr.dyn_cast<StringAttr>()) {
    os << sAttr.getValue();
    return success();
  }
  if (auto sAttr = attr.dyn_cast<SymbolRefAttr>()) {
    if (sAttr.getNestedReferences().size() > 1) {
      return op.emitError(" attribute has more than 1 nested reference");
    }
    os << sAttr.getRootReference();
    return success();
  }
  if (auto type = attr.dyn_cast<TypeAttr>()) {
    return emitType(op, type.getValue());
  }
  return op.emitError("cannot emit attribute of type ") << attr.getType();
}

LogicalResult CppEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitOpError() << "operand value not in scope";
    os << getOrCreateName(result);
    return success();
  };
  return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

LogicalResult
CppEmitter::emitOperandsAndAttributes(Operation &op,
                                      ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (auto attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.first.strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.first.strref()))
      return success();
    os << "/* " << attr.first << " */";
    if (failed(emitAttribute(op, attr.second)))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult CppEmitter::emitVariableAssignment(OpResult result) {
  if (!hasValueInScope(result)) {
    return result.getDefiningOp()->emitOpError(
        "result variable for the operation has not been declared.");
  }
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result,
                                                  bool trailingSemicolon) {
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared.");
  }
  if (failed(emitType(*result.getOwner(), result.getType()))) {
    return failure();
  }
  os << " " << getOrCreateName(result);
  if (trailingSemicolon) {
    os << ";\n";
  }
  return success();
}

LogicalResult CppEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    auto result = op.getResult(0);
    if (forwardDeclaredVariables()) {
      if (failed(emitVariableAssignment(result)))
        return failure();
    } else {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
        return failure();
      os << " = ";
    }
    break;
  }
  default:
    if (!forwardDeclaredVariables()) {
      for (auto result : op.getResults()) {
        if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
          return failure();
      }
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult CppEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block)) {
    return block.getParentOp()->emitError("Label for block not found.");
  }
  os << getOrCreateName(block) << ":\n";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, Operation &op) {
  // EmitC ops.
  if (auto applyOp = dyn_cast<emitc::ApplyOp>(op))
    return printApplyOp(emitter, applyOp);
  if (auto callOp = dyn_cast<emitc::CallOp>(op))
    return printCallOp(emitter, callOp);
  if (auto constOp = dyn_cast<emitc::ConstOp>(op))
    return printConstantOp(emitter, constOp);
  if (auto forOp = dyn_cast<emitc::ForOp>(op))
    return printForOp(emitter, forOp);
  if (auto ifOp = dyn_cast<emitc::IfOp>(op))
    return printIfOp(emitter, ifOp);
  if (auto yieldOp = dyn_cast<emitc::YieldOp>(op))
    return printYieldOp(emitter, yieldOp);

  // Standard ops.
  if (auto branchOp = dyn_cast<BranchOp>(op))
    return printBranchOp(emitter, branchOp);
  if (auto callOp = dyn_cast<mlir::CallOp>(op))
    return printCallOp(emitter, callOp);
  if (auto branchOp = dyn_cast<CondBranchOp>(op))
    return printCondBranchOp(emitter, branchOp);
  if (auto constantOp = dyn_cast<ConstantOp>(op))
    return printConstantOp(emitter, constantOp);
  if (auto funcOp = dyn_cast<FuncOp>(op))
    return printFunction(emitter, funcOp);
  if (auto moduleOp = dyn_cast<ModuleOp>(op))
    return printModule(emitter, moduleOp);
  if (auto returnOp = dyn_cast<ReturnOp>(op))
    return printReturnOp(emitter, returnOp);

  return op.emitOpError() << "unable to find printer for op";
}

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  if (failed(printOperation(*this, op)))
    return failure();
  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult CppEmitter::emitType(Operation &op, Type type) {
  if (auto iType = type.dyn_cast<IntegerType>()) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (mapToSigned(iType.getSignedness())) {
        return (os << "int" << iType.getWidth() << "_t"), success();
      } else {
        return (os << "uint" << iType.getWidth() << "_t"), success();
      }
    default:
      return op.emitError("cannot emit integer type ") << type;
    }
  }
  if (auto fType = type.dyn_cast<FloatType>()) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return op.emitError("cannot emit float type ") << type;
    }
  }
  if (auto iType = type.dyn_cast<IndexType>()) {
    return (os << "size_t"), success();
  }
  if (auto tType = type.dyn_cast<TensorType>()) {
    // TensorType is not supported if emitting C.
    if (restrictedToC())
      return op.emitError("cannot emit tensor type if emitting C");
    if (!tType.hasRank())
      return op.emitError("cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return op.emitError("cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (failed(emitType(op, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }
  if (auto tType = type.dyn_cast<TupleType>()) {
    return emitTupleType(op, tType.getTypes());
  }
  if (auto oType = type.dyn_cast<emitc::OpaqueType>()) {
    os << oType.getValue();
    return success();
  }
  return op.emitError("cannot emit type ") << type;
}

LogicalResult CppEmitter::emitTypes(Operation &op, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(op, types.front());
  default:
    return emitTupleType(op, types);
  }
}

LogicalResult CppEmitter::emitTupleType(Operation &op, ArrayRef<Type> types) {
  if (restrictedToC())
    return op.emitError("cannot emit tuple type if emitting C");
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(op, type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult emitc::TranslateToCpp(Operation &op, TargetOptions targetOptions,
                                    raw_ostream &os, bool trailingSemicolon) {
  CppEmitter emitter(
      os, /*restrictToC=*/false,
      /*forwardDeclareVariables=*/targetOptions.forwardDeclareVariables);
  return emitter.emitOperation(op, trailingSemicolon);
}

LogicalResult emitc::TranslateToC(Operation &op, TargetOptions targetOptions,
                                  raw_ostream &os, bool trailingSemicolon) {
  CppEmitter emitter(
      os, /*restrictToC=*/true,
      /*forwardDeclareVariables=*/targetOptions.forwardDeclareVariables);
  return emitter.emitOperation(op, trailingSemicolon);
}
