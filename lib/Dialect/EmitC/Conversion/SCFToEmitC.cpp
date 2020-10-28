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

namespace {

class ForOpConversion : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto forOp = rewriter.create<emitc::ForOp>(op.getLoc(), op.lowerBound(),
                                               op.upperBound(), op.step(),
                                               op.initArgs());
    rewriter.eraseBlock(forOp.getBody());
    rewriter.inlineRegionBefore(op.region(), forOp.region(),
                                forOp.region().end());
    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }
};

class IfOpConversion : public OpRewritePattern<scf::IfOp> {
public:
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    bool hasElseRegion = !op.elseRegion().empty();
    auto ifOp = rewriter.create<emitc::IfOp>(op.getLoc(), op.getResultTypes(),
                                             op.condition(), hasElseRegion);
    rewriter.inlineRegionBefore(op.thenRegion(), &ifOp.thenRegion().back());
    rewriter.eraseBlock(&ifOp.thenRegion().back());
    if (hasElseRegion) {
      rewriter.inlineRegionBefore(op.elseRegion(), &ifOp.elseRegion().back());
      rewriter.eraseBlock(&ifOp.elseRegion().back());
    }

    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

class YieldOpConversion : public OpRewritePattern<scf::YieldOp> {
public:
  using OpRewritePattern<scf::YieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::YieldOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::YieldOp>(op, op.getOperands());
    return success();
  }
};

} // namespace

void populateScfToEmitcPatterns(MLIRContext *ctx,
                                OwningRewritePatternList &patterns) {
  patterns.insert<ForOpConversion>(ctx);
  patterns.insert<IfOpConversion>(ctx);
  patterns.insert<YieldOpConversion>(ctx);
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
    target.addIllegalOp<scf::ForOp, scf::IfOp, scf::YieldOp>();

    OwningRewritePatternList patterns;
    populateScfToEmitcPatterns(&getContext(), patterns);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
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
