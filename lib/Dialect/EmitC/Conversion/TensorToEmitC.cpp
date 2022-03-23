//===- TensorToEmitC.cpp - Tensor to EmitC conversion ---------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for converting tensor to the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "emitc/Dialect/EmitC/Conversion/Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::emitc;

namespace {

/// Convert `tensor.extract` into an `emitc.call` operation.
class ExtractOpConversion : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;

public:
  ExtractOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tensor::ExtractOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tensor::ExtractOp indexCastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::tensor::extract");

    Type elementType = indexCastOp.getType();
    if (auto tensorType = elementType.dyn_cast<TensorType>()) {
      elementType = tensorType.getElementType();
    }

    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        indexCastOp, indexCastOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};

/// Convert `tensor.splat` into an `emitc.call` operation.
class SplatOpConversion : public OpConversionPattern<tensor::SplatOp> {
  using OpConversionPattern<tensor::SplatOp>::OpConversionPattern;

public:
  SplatOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tensor::SplatOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tensor::SplatOp splatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::tensor::splat");

    ArrayAttr args;
    Type resultType = splatOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(splatOp, splatOp.getType(),
                                               callee, args, templateArgs,
                                               adaptor.getOperands());

    return success();
  }
};
} // namespace

void populateTensorToEmitcPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns) {
  patterns.add<ExtractOpConversion>(ctx);
  patterns.add<SplatOpConversion>(ctx);
}

namespace {

struct ConvertTensorToEmitCPass
    : public ConvertTensorToEmitCBase<ConvertTensorToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnOperation() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalOp<tensor::ExtractOp>();
    target.addIllegalOp<tensor::SplatOp>();

    RewritePatternSet patterns(&getContext());
    populateTensorToEmitcPatterns(&getContext(), patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::emitc::createConvertTensorToEmitCPass() {
  return std::make_unique<ConvertTensorToEmitCPass>();
}
