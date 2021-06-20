//===- EmitCInsertInclude.cpp - Insert EmitC includes -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for inserting EmitC IncludeOps.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "emitc/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitc {

namespace {

void insertIncludeOp(ModuleOp &op, StringRef includeName) {
  OpBuilder builder(op);

  builder.setInsertionPointToStart(&op.getRegion().getBlocks().front());
  builder.create<emitc::IncludeOp>(op.getLoc(), includeName, false);
}

struct InsertEmitCMHLOIncludePass
    : public InsertEmitCMHLOIncludeBase<InsertEmitCMHLOIncludePass> {
  void runOnOperation() override {
    auto op = getOperation();
    insertIncludeOp(op, "emitc_mhlo.h");
  }
};

struct InsertEmitCStdIncludePass
    : public InsertEmitCStdIncludeBase<InsertEmitCStdIncludePass> {
  void runOnOperation() override {
    auto op = getOperation();
    insertIncludeOp(op, "emitc_std.h");
  }
};

struct InsertEmitCTensorIncludePass
    : public InsertEmitCTensorIncludeBase<InsertEmitCTensorIncludePass> {
  void runOnOperation() override {
    auto op = getOperation();
    insertIncludeOp(op, "emitc_tensor.h");
  }
};

struct InsertEmitCTosaIncludePass
    : public InsertEmitCTosaIncludeBase<InsertEmitCTosaIncludePass> {
  void runOnOperation() override {
    auto op = getOperation();
    insertIncludeOp(op, "emitc_tosa.h");
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createInsertEmitCMHLOIncludePass() {
  return std::make_unique<InsertEmitCMHLOIncludePass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createInsertEmitCStdIncludePass() {
  return std::make_unique<InsertEmitCStdIncludePass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createInsertEmitCTensorIncludePass() {
  return std::make_unique<InsertEmitCTensorIncludePass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createInsertEmitCTosaIncludePass() {
  return std::make_unique<InsertEmitCTosaIncludePass>();
}

} // namespace emitc
} // namespace mlir
