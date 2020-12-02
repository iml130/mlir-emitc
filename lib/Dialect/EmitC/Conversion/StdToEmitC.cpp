//===- StdToEmitC.cpp - std to EmitC conversion ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for lowering std dialect to EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "emitc/Dialect/EmitC/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitc {

namespace {
class ExtractElementOpConversion
    : public OpConversionPattern<ExtractElementOp> {
  using OpConversionPattern<ExtractElementOp>::OpConversionPattern;

public:
  ExtractElementOpConversion(MLIRContext *ctx)
      : OpConversionPattern<ExtractElementOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(ExtractElementOp indexCastOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("standard::extract_element");

    Type elementType = indexCastOp.getType();
    if (auto tensorType = elementType.dyn_cast<TensorType>()) {
      elementType = tensorType.getElementType();
    }

    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(indexCastOp,
                                               indexCastOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }
};

class IndexCastOpConversion : public OpConversionPattern<IndexCastOp> {
  using OpConversionPattern<IndexCastOp>::OpConversionPattern;

public:
  IndexCastOpConversion(MLIRContext *ctx)
      : OpConversionPattern<IndexCastOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(IndexCastOp indexCastOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("standard::index_cast");

    ArrayAttr args;
    Type resultType = indexCastOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(indexCastOp,
                                               indexCastOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }
};

class SplatOpConversion : public OpConversionPattern<SplatOp> {
  using OpConversionPattern<SplatOp>::OpConversionPattern;

public:
  SplatOpConversion(MLIRContext *ctx) : OpConversionPattern<SplatOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(SplatOp splatOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("standard::splat");

    ArrayAttr args;
    Type resultType = splatOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        splatOp, splatOp.getType(), callee, args, templateArgs, operands);

    return success();
  }
};
} // namespace

void populateStdToEmitcPatterns(MLIRContext *ctx,
                                OwningRewritePatternList &patterns) {
  patterns.insert<ExtractElementOpConversion>(ctx);
  patterns.insert<IndexCastOpConversion>(ctx);
  patterns.insert<SplatOpConversion>(ctx);
}

namespace {

struct ConvertStdToEmitCPass
    : public ConvertStdToEmitCBase<ConvertStdToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addIllegalOp<ExtractElementOp>();
    target.addIllegalOp<IndexCastOp>();
    target.addIllegalOp<SplatOp>();

    OwningRewritePatternList patterns;
    populateStdToEmitcPatterns(&getContext(), patterns);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<FunctionPass> createConvertStdToEmitCPass() {
  return std::make_unique<ConvertStdToEmitCPass>();
}

} // namespace emitc
} // namespace mlir
