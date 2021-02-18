//===- TosaToEmitC.cpp - TOSA to EmitC conversion ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for converting TOSA to the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "emitc/Dialect/EmitC/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitc {

namespace {

template <typename SrcOp>
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
  matchAndRewrite(SrcOp srcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args;

    SmallVector<Attribute, 4> templateArgs_;

    if (explicitResultType) {
      Type type = srcOp.getType();
      templateArgs_.push_back(TypeAttr::get(type));
    }

    if (explicitOperandTypes) {
      for (auto &operand : operands) {
        Type type = operand.getType();
        templateArgs_.push_back(TypeAttr::get(type));
      }
    }

    ArrayAttr
        templateArgs; // = ArrayAttr::get(templateArgs_, srcOp.getContext());

    rewriter.replaceOpWithNewOp<emitc::CallOp>(srcOp, srcOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }

  StringRef funcName;
  // If set, use the result type of the operation as template parameter
  bool explicitResultType;
  // If set, use the operand types as (additional) template parameters
  bool explicitOperandTypes;
};

} // namespace

void populateTosaToEmitcPatterns(MLIRContext *ctx,
                                 OwningRewritePatternList &patterns) {
  patterns.insert<CallOpConversion<tosa::AbsOp>>(ctx, "tosa::abs");
}

namespace {

struct ConvertTosaToEmitCPass
    : public ConvertTosaToEmitCBase<ConvertTosaToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {
    // Convert other ops
    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<tosa::TosaDialect>();
    // TODO: We might to readd further legal dialects and ops
    // target.addLegalDialect<StandardOpsDialect>();
    // target.addLegalOp<FuncOp>();
    // target.addLegalOp<ModuleOp>();
    // target.addLegalOp<ModuleTerminatorOp>();

    target.addIllegalOp<tosa::AbsOp>();

    OwningRewritePatternList patterns;
    populateTosaToEmitcPatterns(&getContext(), patterns);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<FunctionPass> createConvertTosaToEmitCPass() {
  return std::make_unique<ConvertTosaToEmitCPass>();
}

} // namespace emitc
} // namespace mlir
