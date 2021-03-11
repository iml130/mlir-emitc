//===- Cpp.h - Helpers to create C++ emitter --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file define a helpers to emit C++ code using the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_TARGET_CPP_H
#define EMITC_TARGET_CPP_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace mlir {
namespace emitc {

struct TargetOptions {
  bool forwardDeclareVariables;
};

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

/// Emitter that uses dialect specific emitters to emit C++ code.
struct CppEmitter {
  explicit CppEmitter(raw_ostream &os, bool restrictToC,
                      bool forwardDeclareVariables);

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

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

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

  /// Whether to map an mlir integer to a signed integer in C++.
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

  /// Returns if to emitc C.
  bool restrictedToC() { return restrictToC; };

  /// Returns if all variables need to be forward declared.
  bool forwardDeclaredVariables() { return forwardDeclareVariables; };

private:
  using ValMapper = llvm::ScopedHashTable<Value, std::string>;

  /// Output stream to emit to.
  raw_ostream &os;

  /// Boolean that restricts the emitter to C.
  bool restrictToC;

  /// Boolean to enforce a forward declaration of all variables.
  bool forwardDeclareVariables;

  /// Map from value to name of C++ variable that contain the name.
  ValMapper mapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
};

/// Translates the given operation to C++ code. The operation or operations in
/// the region of 'op' need almost all be in EmitC dialect.
LogicalResult TranslateToCpp(Operation &op, TargetOptions targetOptions,
                             raw_ostream &os, bool trailingSemicolon = false);

/// Similar to `TranslateToCpp`, but translates the given operation to C code.
LogicalResult TranslateToC(Operation &op, TargetOptions targetOptions,
                           raw_ostream &os, bool trailingSemicolon = false);
} // namespace emitc
} // namespace mlir

#endif // EMITC_TARGET_CPP_H
