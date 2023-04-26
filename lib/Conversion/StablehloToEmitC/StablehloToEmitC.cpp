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
#include "emitc/Conversion/StablehloToEmitC/StablehloToEmitC.h"

using namespace mlir;
using namespace mlir::emitc;

namespace {

/// Convert a common `stablehlo` operation into an `emitc.call` operation.
template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class CallOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  CallOpConversion(MLIRContext *ctx, StringRef funcName,
                   bool explicitResultType = false,
                   bool explicitOperandTypes = false)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName),
        explicitResultType(explicitResultType),
        explicitOperandTypes(explicitOperandTypes) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args;

    SmallVector<Attribute, 4> templateArguments;

    if (explicitResultType) {
      Type type = srcOp.getType();
      templateArguments.push_back(TypeAttr::get(type));
    }

    if (explicitOperandTypes) {
      for (auto operand : adaptor.getOperands()) {
        Type type = operand.getType();
        templateArguments.push_back(TypeAttr::get(type));
      }
    }
    ArrayAttr templateArgs;
    if (!templateArguments.empty()) {
      templateArgs = ArrayAttr::get(srcOp.getContext(), templateArguments);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(srcOp, srcOp.getType(), callee,
                                               args, templateArgs,
                                               adaptor.getOperands());

    return success();
  }

  StringRef funcName;
  // If set, use the result type of the operation as template parameter.
  bool explicitResultType;
  // If set, use the operand types as (additional) template parameters.
  bool explicitOperandTypes;
};

} // namespace

void populateStablehloToEmitcPatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns) {
  // Insert patterns for StableHLO unary elementwise ops.
  patterns.add<CallOpConversion<stablehlo::AbsOp>>(ctx, "emitc::mhlo::abs");
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
