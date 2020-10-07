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
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <stack>

#define DEBUG_TYPE "translate-to-cpp"

using namespace mlir;
using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STL as functions used on
/// each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor each_fn, NullaryFunctor between_fn) {
  if (begin == end)
    return success();
  if (failed(each_fn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    between_fn();
    if (failed(each_fn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor each_fn,
                                         NullaryFunctor between_fn) {
  return interleaveWithError(c.begin(), c.end(), each_fn, between_fn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor each_fn) {
  return interleaveWithError(c.begin(), c.end(), each_fn,
                             [&]() { os << ", "; });
}

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct CppEmitter {
  explicit CppEmitter(raw_ostream &os);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon = true);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  LogicalResult emitTypes(ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(ArrayRef<Type> types);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Whether to map an mlir integer to a signed integer in C++
  bool mapToSigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(CppEmitter &emitter) : mapperScope(emitter.mapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
    }
    ~Scope() { emitter.valueInScopeCount.pop(); }

  private:
    llvm::ScopedHashTableScope<Value, std::string> mapperScope;
    CppEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  /// Returns the output stream.
  raw_ostream &ostream() { return os; };

private:
  using ValMapper = llvm::ScopedHashTable<Value, std::string>;

  /// Output stream to emit to.
  raw_ostream &os;

  /// Map from value to name of C++ variable that contain the name.
  ValMapper mapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
};
} // namespace

static LogicalResult printConstantOp(CppEmitter &emitter,
                                     ConstantOp constantOp) {
  auto &os = emitter.ostream();
  emitter.emitType(constantOp.getType());
  os << " " << emitter.getOrCreateName(constantOp.getResult());
  if (failed(emitter.emitAttribute(constantOp.getValue())))
    return constantOp.emitError("unable to emit constant value");
  return success();
}

static LogicalResult printCallOp(CppEmitter &emitter, CallOp callOp) {
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
    return emitter.emitAttribute(attr);
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

static LogicalResult printForOp(CppEmitter &emitter, emitc::ForOp forOp) {
  auto &os = emitter.ostream();

  if (forOp.getNumRegionIterArgs() != 0) {
    auto regionArgs = forOp.getRegionIterArgs();
    auto operands = forOp.getIterOperands();

    for (auto i : llvm::zip(regionArgs, operands)) {
      emitter.emitType(std::get<0>(i).getType());
      os << " " << emitter.getOrCreateName(std::get<0>(i)) << " = ";
      os << emitter.getOrCreateName(std::get<1>(i)) << ";";
      os << "\n";
    }
  }

  if (forOp.getNumResults() != 0) {
    for (auto op : forOp.getResults()) {
      emitter.emitType(op.getType());
      os << " " << emitter.getOrCreateName(op) << ";";
      os << "\n";
    }
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
    emitter.emitOperation(op);
  }

  os << "}\n";
  return success();
}

static LogicalResult printIfOp(CppEmitter &emitter, emitc::IfOp ifOp) {
  auto &os = emitter.ostream();

  if (ifOp.getNumResults() != 0) {
    for (auto op : ifOp.getResults()) {
      emitter.emitType(op.getType());
      os << " " << emitter.getOrCreateName(op) << ";";
      os << "\n";
    }
  }

  os << "if (";
  if (failed(emitter.emitOperands(*ifOp.getOperation())))
    return failure();
  os << ") {\n";

  auto &thenRegion = ifOp.thenRegion();
  for (auto &op : thenRegion.getOps()) {
    emitter.emitOperation(op);
  }

  os << "}\n";

  auto &elseRegion = ifOp.elseRegion();
  if (!elseRegion.empty()) {
    os << "else {\n";

    for (auto &op : elseRegion.getOps()) {
      emitter.emitOperation(op);
    }

    os << "}\n";
  }

  return success();
}

static LogicalResult printYieldOp(CppEmitter &emitter, emitc::YieldOp yieldOp) {
  auto &os = emitter.ostream();

  if (yieldOp.getNumOperands() == 0) {
    return success();
  } else {
    auto &parentOp = *yieldOp.getParentOp();

    for (uint result = 0; result < parentOp.getNumResults(); ++result) {
      os << emitter.getOrCreateName(parentOp.getResult(result)) << " = ";

      if (!emitter.hasValueInScope(yieldOp.getOperand(result)))
        return yieldOp.emitError() << "operand value not in scope";
      os << emitter.getOrCreateName(yieldOp.getOperand(result)) << ";\n";
    }
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
  os << "#include <cmath>\n\n";
  os << "#include \"emitc_mhlo.h\"\n";
  os << "#include \"emitc_std.h\"\n\n";
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
    if (failed(emitter.emitOperation(op, /*trailingSemiColon=*/false)))
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

CppEmitter::CppEmitter(raw_ostream &os) : os(os) { valueInScopeCount.push(0); }

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

LogicalResult CppEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    auto result = op.getResult(0);
    if (failed(emitType(result.getType())))
      return op.emitError() << "unable to emit type " << result.getType();
    os << " " << getOrCreateName(result) << " = ";
    break;
  }
  default:
    for (auto result : op.getResults()) {
      if (failed(emitType(result.getType())))
        return failure();
      os << " " << getOrCreateName(result) << ";\n";
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, Operation &op) {
  if (auto callOp = dyn_cast<CallOp>(op))
    return printCallOp(emitter, callOp);
  if (auto callOp = dyn_cast<emitc::CallOp>(op))
    return printCallOp(emitter, callOp);
  if (auto ifOp = dyn_cast<emitc::IfOp>(op))
    return printIfOp(emitter, ifOp);
  if (auto yieldOp = dyn_cast<emitc::YieldOp>(op))
    return printYieldOp(emitter, yieldOp);
  if (auto forOp = dyn_cast<emitc::ForOp>(op))
    return printForOp(emitter, forOp);
  if (auto constantOp = dyn_cast<ConstantOp>(op))
    return printConstantOp(emitter, constantOp);
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
  if (auto itype = type.dyn_cast<IntegerType>()) {
    switch (itype.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (mapToSigned(itype.getSignedness())) {
        return (os << "int" << itype.getWidth() << "_t"), success();
      } else {
        return (os << "uint" << itype.getWidth() << "_t"), success();
      }
    default:
      return failure();
    }
  }
  if (auto itype = type.dyn_cast<FloatType>()) {
    switch (itype.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return failure();
    }
  }
  if (auto itype = type.dyn_cast<IndexType>()) {
    return (os << "size_t"), success();
  }
  if (auto itype = type.dyn_cast<TensorType>()) {
    if (!itype.hasRank())
      return failure();
    os << "Tensor";
    os << itype.getRank();
    os << "D<";
    emitType(itype.getElementType());
    auto shape = itype.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }
  if (auto ttype = type.dyn_cast<TupleType>()) {
    return emitTupleType(ttype.getTypes());
  }
  // TODO: Change to be EmitC specific.
  if (auto ot = type.dyn_cast<OpaqueType>()) {
    os << ot.getTypeData();
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
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult emitc::TranslateToCpp(Operation &op, raw_ostream &os,
                                    bool trailingSemicolon) {
  CppEmitter emitter(os);
  return emitter.emitOperation(op, trailingSemicolon);
}
