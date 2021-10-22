//===- AirthToEmitC.cpp - std to EmitC conversion -------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for lowering arith dialect to EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "emitc/Dialect/EmitC/Conversion/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::emitc;

namespace {

/// Convert `arith.constant` into an `emitc.constant` operation.
class ConstantOpConversion : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        constantOp, constantOp.getType(), constantOp.value());
    return success();
  }
};

// Convert `arith.index_cast` into an `emitc.call` operation.
class IndexCastOpConversion : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

public:
  IndexCastOpConversion(MLIRContext *ctx)
      : OpConversionPattern<arith::IndexCastOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(arith::IndexCastOp indexCastOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::arith::index_cast");

    ArrayAttr args;
    Type resultType = indexCastOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(indexCastOp,
                                               indexCastOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }
};
} // namespace

void populateArithToEmitcPatterns(MLIRContext *ctx,
                                  RewritePatternSet &patterns) {
  patterns.add<ConstantOpConversion>(ctx);
  patterns.add<IndexCastOpConversion>(ctx);
}

namespace {

struct ConvertArithToEmitCPass
    : public ConvertArithToEmitCBase<ConvertArithToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addIllegalOp<arith::ConstantOp>();
    target.addIllegalOp<arith::IndexCastOp>();

    RewritePatternSet patterns(&getContext());
    populateArithToEmitcPatterns(&getContext(), patterns);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<FunctionPass> mlir::emitc::createConvertArithToEmitCPass() {
  return std::make_unique<ConvertArithToEmitCPass>();
}
