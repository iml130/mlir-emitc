//===- SCFToEmitC.cpp - SCF to EmitC conversion ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for lowering SCF dialect to EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "emitc/Dialect/EmitC/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitc {

void populateScfToEmitcPatterns(MLIRContext *ctx,
                                 OwningRewritePatternList &patterns) {
  return;
}

namespace {

struct ConvertScfToEmitcPass
    : public PassWrapper<ConvertScfToEmitcPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect>();
  }
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalOp<scf::ForOp, scf::IfOp>();

    OwningRewritePatternList patterns;
    //populateScfToEmitcPatterns(&getContext(), patterns);

    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createConvertScfToEmitcPass() {
  return std::make_unique<ConvertScfToEmitcPass>();
}

} // namespace emitc
} // namespace mlir
