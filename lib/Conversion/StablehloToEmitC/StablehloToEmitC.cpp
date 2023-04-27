//===- StablehloToEmitC.cpp - StableHLO to EmitC conversion ---------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for lowering StableHLO dialect to EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "../PassDetail.h"
#include "emitc/Conversion/EmitCCommon/GenericOpConversion.h"
#include "emitc/Conversion/StablehloToEmitC/StablehloToEmitC.h"

using namespace mlir;
using namespace mlir::emitc;

void populateStablehloToEmitcPatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns) {
  // Insert patterns for StableHLO unary elementwise ops.
  patterns.add<GenericOpConversion<stablehlo::AbsOp>>(ctx, "emitc::mhlo::abs");
}

namespace {

struct ConvertStablehloToEmitCPass
    : public ConvertStablehloToEmitCBase<ConvertStablehloToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnOperation() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();

    // clang-format off
    // StableHLO unary elementwise ops
    target.addIllegalOp<stablehlo::AbsOp>();
    // clang-format on

    RewritePatternSet patterns(&getContext());
    populateStablehloToEmitcPatterns(&getContext(), patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::emitc::createConvertStablehloToEmitCPass() {
  return std::make_unique<ConvertStablehloToEmitCPass>();
}
