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

static LogicalResult printConstantOp(CppEmitter &emitter,
                                     ConstantOp constantOp) {
  auto &os = emitter.ostream();
  if (failed(emitter.emitVariableDeclaration(
          constantOp.getOperation()->getResult(0),
          /*trailingSemicolon=*/false)))
    return failure();

  // Add braces for number literals only to avoid double brace intialization for
  // tensors.
  auto value = constantOp.getValue();
  bool emitBraces = value.isa<FloatAttr>() || value.isa<IntegerAttr>();

  // Replace curly braces with `=` if restricted to C
  if (emitter.restrictedToC() && emitBraces) {
    emitBraces = false;
    os << " = ";
  }

  if (emitBraces)
    os << "{";

  if (failed(emitter.emitAttribute(constantOp.getValue())))
    return constantOp.emitError("unable to emit constant value");

  if (emitBraces)
    os << "}";

  return success();
}

static LogicalResult printConstOp(CppEmitter &emitter, ConstOp constOp) {
  auto &os = emitter.ostream();
  if (failed(
          emitter.emitVariableDeclaration(constOp.getOperation()->getResult(0),
                                          /*trailingSemicolon=*/false)))
    return failure();

  // Add braces for number literals only to avoid double brace intialization for
  // tensors.
  auto value = constOp.value();
  bool emitBraces = value.isa<FloatAttr>() || value.isa<IntegerAttr>();

  // Add braces for non empty StringAttr
  if (auto sAttr = value.dyn_cast<StringAttr>()) {
    if (sAttr.getValue().empty()) {
      emitBraces = false;
    } else {
      emitBraces = true;
    }
  }

  // Replace curly braces with `=` if restricted to C
  if (emitter.restrictedToC() && emitBraces) {
    emitBraces = false;
    os << " = ";
  }

  if (emitBraces)
    os << "{";

  if (failed(emitter.emitAttribute(value)))
    return constOp.emitError("unable to emit constant value");

  if (emitBraces)
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
    if (failed(emitter.emitAttribute(attr))) {
      return op.emitError() << "unable to emit attribute " << attr;
    }
    return success();
  };

  if (callOp.template_args()) {
    os << "<";
    if (failed(interleaveCommaWithError(*callOp.template_args(), os, emitArgs)))
      return failure();
    os << ">";
  }

  os << "(";

  // if (callOp.argsAttr()) {
  //  callOp.dump();
  //}
  auto emittedArgs =
      callOp.args() ? interleaveCommaWithError(*callOp.args(), os, emitArgs)
                    : emitter.emitOperands(op);
  if (failed(emittedArgs))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printGetAddressOfOp(CppEmitter &emitter,
                                         emitc::GetAddressOfOp getAddressOfOp) {
  auto &os = emitter.ostream();
  auto &op = *getAddressOfOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << "&";
  os << emitter.getOrCreateName(getAddressOfOp.getOperand());

  return success();
}

static LogicalResult printForOp(CppEmitter &emitter, emitc::ForOp forOp) {
  auto &os = emitter.ostream();

  if (forOp.getNumRegionIterArgs() != 0) {
    auto regionArgs = forOp.getRegionIterArgs();
    auto operands = forOp.getIterOperands();

    for (auto i : llvm::zip(regionArgs, operands)) {
      if (failed(emitter.emitType(std::get<0>(i).getType())))
        return failure();
      os << " " << emitter.getOrCreateName(std::get<0>(i)) << " = ";
      os << emitter.getOrCreateName(std::get<1>(i)) << ";";
      os << "\n";
    }
  }

  for (auto result : forOp.getResults()) {
    if (failed(emitter.emitVariableDeclaration(result,
                                               /*trailingSemicolon=*/true)))
      return failure();
  }

  os << "for (";
  if (failed(emitter.emitType(forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << "=";
  os << emitter.getOrCreateName(forOp.lowerBound());
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << "<";
  os << emitter.getOrCreateName(forOp.upperBound());
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << "=";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << "+";
  os << emitter.getOrCreateName(forOp.step());
  os << ") {\n";

  auto &forRegion = forOp.region();
  for (auto &op : forRegion.getOps()) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }

  os << "}\n";
  return success();
}

static LogicalResult printIfOp(CppEmitter &emitter, emitc::IfOp ifOp) {
  auto &os = emitter.ostream();

  for (auto result : ifOp.getResults()) {
    if (failed(emitter.emitVariableDeclaration(result,
                                               /*trailingSemicolon=*/true)))
      return failure();
  }

  os << "if (";
  if (failed(emitter.emitOperands(*ifOp.getOperation())))
    return failure();
  os << ") {\n";

  auto &thenRegion = ifOp.thenRegion();
  for (auto &op : thenRegion.getOps()) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }

  os << "}\n";

  auto &elseRegion = ifOp.elseRegion();
  if (!elseRegion.empty()) {
    os << "else {\n";

    for (auto &op : elseRegion.getOps()) {
      if (failed(emitter.emitOperation(op)))
        return failure();
    }

    os << "}\n";
  }

  return success();
}

static LogicalResult printYieldOp(CppEmitter &emitter, emitc::YieldOp yieldOp) {
  auto &os = emitter.ostream();
  auto &parentOp = *yieldOp->getParentOp();

  if (yieldOp.getNumOperands() != parentOp.getNumResults()) {
    return failure();
  }

  for (auto pair : llvm::zip(parentOp.getResults(), yieldOp.getOperands())) {
    auto result = std::get<0>(pair);
    auto operand = std::get<1>(pair);
    os << emitter.getOrCreateName(result) << " = ";

    if (!emitter.hasValueInScope(operand))
      return yieldOp.emitError() << "operand value not in scope";
    os << emitter.getOrCreateName(operand) << ";\n";
  }

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
    if (failed(emitter.emitTypes(funcOp.getType().getResults())))
      return funcOp.emitError() << "failed to convert operand type";
    os << " " << funcOp.getName() << "(";
    if (failed(interleaveCommaWithError(
            funcOp.getArguments(), os, [&](BlockArgument arg) {
              return emitter.emitType(arg.getType());
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
  auto &blocks = functionOp.getBlocks();
  if (blocks.size() != 1)
    return functionOp.emitOpError() << "only single block functions supported";

  CppEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();
  if (failed(emitter.emitTypes(functionOp.getType().getResults())))
    return functionOp.emitError() << "unable to emit all types";
  os << " " << functionOp.getName();

  os << "(";
  if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
            if (failed(emitter.emitType(arg.getType())))
              return functionOp.emitError() << "unable to emit arg "
                                            << arg.getArgNumber() << "'s type";
            os << " " << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ") {\n";

  for (Operation &op : functionOp.front()) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }
  os << "}\n";
  return success();
}

CppEmitter::CppEmitter(raw_ostream &os, bool restrictToC,
                       bool forwardDeclareVariables)
    : os(os), restrictToC(restrictToC),
      forwardDeclareVariables(forwardDeclareVariables) {
  valueInScopeCount.push(0);
}

/// Return the existing or a new name for a Value*.
StringRef CppEmitter::getOrCreateName(Value val) {
  if (!mapper.count(val))
    mapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *mapper.begin(val);
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

bool CppEmitter::hasValueInScope(Value val) { return mapper.count(val); }

LogicalResult CppEmitter::emitAttribute(Attribute attr) {

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
  if (auto dense = attr.dyn_cast<mlir::DenseFPElementsAttr>()) {
    // Dense attributes are not supported if emitting C
    if (restrictedToC())
      return failure();
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
    // Dense attributes are not supported if emitting C
    if (restrictedToC())
      return failure();
    if (auto iType = dense.getType()
                         .cast<TensorType>()
                         .getElementType()
                         .cast<IntegerType>()) {
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
                         .cast<IndexType>()) {
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
      return failure();
    }
    os << sAttr.getRootReference();
    return success();
  }
  if (auto type = attr.dyn_cast<TypeAttr>()) {
    return emitType(type.getValue());
  }
  return failure();
}

LogicalResult CppEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitError() << "operand value not in scope";
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
    if (failed(emitAttribute(attr.second)))
      return op.emitError() << "unable to emit attribute " << attr.second;
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result,
                                                  bool trailingSemicolon) {
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared.");
  }
  if (failed(emitType(result.getType()))) {
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
    if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
      return op.emitError() << "unable to emit type " << result.getType();
    os << " = ";
    break;
  }
  default:
    for (auto result : op.getResults()) {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
        return failure();
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, Operation &op) {
  if (auto callOp = dyn_cast<mlir::CallOp>(op))
    return printCallOp(emitter, callOp);
  if (auto callOp = dyn_cast<emitc::CallOp>(op))
    return printCallOp(emitter, callOp);
  if (auto getAdressOfOp = dyn_cast<emitc::GetAddressOfOp>(op))
    return printGetAddressOfOp(emitter, getAdressOfOp);
  if (auto ifOp = dyn_cast<emitc::IfOp>(op))
    return printIfOp(emitter, ifOp);
  if (auto yieldOp = dyn_cast<emitc::YieldOp>(op))
    return printYieldOp(emitter, yieldOp);
  if (auto forOp = dyn_cast<emitc::ForOp>(op))
    return printForOp(emitter, forOp);
  if (auto constantOp = dyn_cast<ConstantOp>(op))
    return printConstantOp(emitter, constantOp);
  if (auto constOp = dyn_cast<emitc::ConstOp>(op))
    return printConstOp(emitter, constOp);
  if (auto returnOp = dyn_cast<ReturnOp>(op))
    return printReturnOp(emitter, returnOp);
  if (auto moduleOp = dyn_cast<ModuleOp>(op))
    return printModule(emitter, moduleOp);
  if (auto funcOp = dyn_cast<FuncOp>(op))
    return printFunction(emitter, funcOp);
  if (isa<ModuleTerminatorOp>(op))
    return success();

  return op.emitOpError() << "unable to find printer for op";
}

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  if (failed(printOperation(*this, op)))
    return failure();
  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult CppEmitter::emitType(Type type) {
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
      return failure();
    }
  }
  if (auto fType = type.dyn_cast<FloatType>()) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return failure();
    }
  }
  if (auto iType = type.dyn_cast<IndexType>()) {
    return (os << "size_t"), success();
  }
  if (auto tType = type.dyn_cast<TensorType>()) {
    // TensorType is not supported if emitting C
    if (restrictedToC())
      return failure();
    if (!tType.hasStaticShape())
      return failure();
    os << "Tensor<";
    if (failed(emitType(tType.getElementType())))
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
    return emitTupleType(tType.getTypes());
  }
  if (auto oType = type.dyn_cast<emitc::OpaqueType>()) {
    os << oType.getValue();
    return success();
  }
  return failure();
}

LogicalResult CppEmitter::emitTypes(ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(types.front());
  default:
    return emitTupleType(types);
  }
}

LogicalResult CppEmitter::emitTupleType(ArrayRef<Type> types) {
  if (restrictedToC())
    return failure();
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(type); })))
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
