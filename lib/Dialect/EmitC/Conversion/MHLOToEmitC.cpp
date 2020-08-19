//===- MHLOToEmitC.cpp - MHLO to EmitC conversion ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for lowering MHLO dialect to EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "emitc/Dialect/EmitC/Passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitc {

namespace {

template <typename SrcOp, typename DstOp>
class UnaryOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  UnaryOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename SrcOp::Adaptor srcAdapter(operands);

    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args =
        rewriter.getArrayAttr({IntegerAttr::get(rewriter.getIndexType(), 0)});

    rewriter.replaceOpWithNewOp<DstOp>(srcOp, srcOp.getType(), callee, args,
                                       operands);

    return success();
  }

  StringRef funcName;
};

// Adopted from IREE's ConvertStandardToVM/ConvertVMToEmitC.
template <typename SrcOp, typename DstOp>
class BinaryOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  BinaryOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename SrcOp::Adaptor srcAdapter(operands);

    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args =
        rewriter.getArrayAttr({IntegerAttr::get(rewriter.getIndexType(), 0),
                               IntegerAttr::get(rewriter.getIndexType(), 1)});
    ValueRange dstOperands{srcAdapter.lhs(), srcAdapter.rhs()};

    rewriter.replaceOpWithNewOp<DstOp>(srcOp, srcAdapter.lhs().getType(),
                                       callee, args, dstOperands);

    return success();
  }

  StringRef funcName;
};

} // namespace

void populateMhloToEmitcPatterns(MLIRContext *ctx,
                                 OwningRewritePatternList &patterns) {
  /// Insert patterns for MHLO unary elementwise ops.
  patterns.insert<UnaryOpConversion<mhlo::AbsOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::abs");
  patterns.insert<UnaryOpConversion<mhlo::CosOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::cos");

  /// Insert patterns for MHLO binary elementwise ops.
  patterns.insert<BinaryOpConversion<mhlo::AddOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::add");
  patterns.insert<BinaryOpConversion<mhlo::DivOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::div");
  patterns.insert<BinaryOpConversion<mhlo::MaxOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::max");
  patterns.insert<BinaryOpConversion<mhlo::MinOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::min");
  patterns.insert<BinaryOpConversion<mhlo::MulOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::mul");
  patterns.insert<BinaryOpConversion<mhlo::PowOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::pow");
  patterns.insert<BinaryOpConversion<mhlo::ShiftLeftOp, emitc::CallOp>>(
      ctx, "mhlo::shift_left");
  patterns.insert<BinaryOpConversion<mhlo::ShiftRightLogicalOp, emitc::CallOp>>(
      ctx, "mhlo::shift_right_logical");
  patterns.insert<BinaryOpConversion<mhlo::SubOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::sub");
}

namespace {

struct ConvertMhloToEmitcPass
    : public PassWrapper<ConvertMhloToEmitcPass, FunctionPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<mhlo::AbsOp, mhlo::CosOp>();
    target.addIllegalOp<mhlo::AddOp, mhlo::DivOp, mhlo::MaxOp, mhlo::MinOp,
                        mhlo::MulOp, mhlo::PowOp, mhlo::ShiftLeftOp,
                        mhlo::ShiftRightLogicalOp, mhlo::SubOp>();

    OwningRewritePatternList patterns;
    populateMhloToEmitcPatterns(&getContext(), patterns);

    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createConvertMhloToEmitcPass() {
  return std::make_unique<ConvertMhloToEmitcPass>();
}

} // namespace emitc
} // namespace mlir
