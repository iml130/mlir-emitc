//===- MHLOToEmitC.cpp - MHLO to EmitC conversion ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for lowering MHLO dialect to EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/Passes.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace emitc {

namespace {
struct ConvertMhloToEmitcPass
    : public PassWrapper<ConvertMhloToEmitcPass, FunctionPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {
      return;
  };
};
} // end anonymous namespace 

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createConvertMhloToEmitcPass() {
  return std::make_unique<ConvertMhloToEmitcPass>();
}

} // namespace emitc
} // namespace mlir
