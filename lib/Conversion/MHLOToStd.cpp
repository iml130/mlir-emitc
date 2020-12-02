//===- MHLOToStd.cpp - MHLO const to std conversion -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for lowering MHLO const op to std.
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/Passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace {
#include "MHLOToStd.inc"
} // namespace
namespace emitc {

void populateMhloToStdPatterns(MLIRContext *ctx,
                               OwningRewritePatternList &patterns) {
  patterns.insert<GeneratedConvert0>(ctx);
}

namespace {

struct ConvertMHLOToStandardPass
    : public PassWrapper<ConvertMHLOToStandardPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect>();
  }
  /// Only lower HLO ConstOp to StandardDialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addIllegalOp<mhlo::ConstOp>();

    OwningRewritePatternList patterns;
    populateMhloToStdPatterns(&getContext(), patterns);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createConvertMHLOToStandardPass() {
  return std::make_unique<ConvertMHLOToStandardPass>();
}

} // namespace emitc
} // namespace mlir
