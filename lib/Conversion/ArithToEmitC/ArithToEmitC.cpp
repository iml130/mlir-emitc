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

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"
#include "emitc/Conversion/ArithToEmitC/ArithToEmitC.h"

using namespace mlir;
using namespace mlir::emitc;

namespace {

// Convert `arith.index_cast` into an `emitc.call_opaque` operation.
class IndexCastOpConversion : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

public:
  IndexCastOpConversion(MLIRContext *ctx)
      : OpConversionPattern<arith::IndexCastOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(arith::IndexCastOp indexCastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::arith::index_cast");

    ArrayAttr args;
    Type resultType = indexCastOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        indexCastOp, indexCastOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};
} // namespace

void populateArithToEmitcPatterns(MLIRContext *ctx,
                                  RewritePatternSet &patterns) {
  patterns.add<IndexCastOpConversion>(ctx);
}

namespace {

struct ConvertArithToEmitCPass
    : public ConvertArithToEmitCBase<ConvertArithToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnOperation() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<arith::IndexCastOp>();

    RewritePatternSet patterns(&getContext());

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    populateArithToEmitcPatterns(&getContext(), patterns);
    populateArithToEmitCPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::emitc::createConvertArithToEmitCPass() {
  return std::make_unique<ConvertArithToEmitCPass>();
}
