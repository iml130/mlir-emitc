//===- MHLOToMHLO.cpp - MHLO to MHLO prerocessing ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic to replace MHLO ops by other MHLO ops.
//
//===----------------------------------------------------------------------===//

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitc {

namespace {

// This function is forked from Google IREE.
// `iree/compiler/Dialect/Flow/Transforms/HLOToHLOPreprocessing.cpp`
class DecomposeLog1PPattern : public OpRewritePattern<mhlo::Log1pOp> {
public:
  using OpRewritePattern<mhlo::Log1pOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::Log1pOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = op.operand().getType().cast<TensorType>();
    DenseElementsAttr attr =
        DenseElementsAttr::get(type, rewriter.getF32FloatAttr(1.0));
    auto one = rewriter.create<ConstantOp>(loc, attr);
    auto x = rewriter.create<mhlo::AddOp>(loc, op.operand(), one);
    rewriter.replaceOpWithNewOp<mhlo::LogOp>(op, x);
    return success();
  }
};

// This function is forked from Google IREE.
// `iree/compiler/Dialect/Flow/Transforms/HLOToHLOPreprocessing.cpp`
class DecomposeExpM1Pattern : public OpRewritePattern<mhlo::Expm1Op> {
public:
  using OpRewritePattern<mhlo::Expm1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::Expm1Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = op.operand().getType().cast<TensorType>();
    DenseElementsAttr attr =
        DenseElementsAttr::get(type, rewriter.getF32FloatAttr(1.0));
    auto one = rewriter.create<ConstantOp>(loc, attr);
    auto x = rewriter.create<mhlo::ExpOp>(loc, op.operand());
    rewriter.replaceOpWithNewOp<mhlo::SubOp>(op, x, one);
    return success();
  }
};

} // namespace

void populateMhloToMhloPatterns(MLIRContext *ctx,
                                OwningRewritePatternList &patterns) {
  patterns.insert<DecomposeLog1PPattern, DecomposeExpM1Pattern>(ctx);
}

namespace {

struct MHLOToMHLOTransform
    : public PassWrapper<MHLOToMHLOTransform, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect, mhlo::MhloDialect>();
  }

  void runOnFunction() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addIllegalOp<mhlo::Log1pOp>();
    target.addIllegalOp<mhlo::Expm1Op>();

    OwningRewritePatternList patterns;
    populateMhloToMhloPatterns(&getContext(), patterns);

    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createMHLOToMHLOPass() {
  return std::make_unique<MHLOToMHLOTransform>();
}

} // namespace emitc
} // namespace mlir
