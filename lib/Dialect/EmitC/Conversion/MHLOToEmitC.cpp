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

class ConcatenateOpConversion
    : public OpConversionPattern<mhlo::ConcatenateOp> {

public:
  ConcatenateOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::ConcatenateOp srcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: Take care of the op's dimension attribute.

    StringRef funcName = "mhlo::concatenate";
    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args =
        rewriter.getArrayAttr({IntegerAttr::get(rewriter.getIndexType(), 0),
                               IntegerAttr::get(rewriter.getIndexType(), 1)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(srcOp, srcOp.getType(), callee,
                                               args, operands);

    return success();
  }
};

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

class TupleOpConversion : public OpConversionPattern<mhlo::TupleOp> {
  using OpConversionPattern<mhlo::TupleOp>::OpConversionPattern;

public:
  TupleOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::TupleOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::TupleOp tupleOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("std::make_tuple");

    // build vector of indices [0 ... operands.size-1]
    std::vector<Attribute> indices;
    for (int i = 0; i < operands.size(); i++) {
      indices.push_back(IntegerAttr::get(rewriter.getIndexType(), i));
    }

    ArrayAttr args = rewriter.getArrayAttr(llvm::makeArrayRef(indices));

    rewriter.replaceOpWithNewOp<emitc::CallOp>(tupleOp, tupleOp.getType(),
                                               callee, args, operands);

    return success();
  }
};

class GetTupleElementOpConversion
    : public OpConversionPattern<mhlo::GetTupleElementOp> {
  using OpConversionPattern<mhlo::GetTupleElementOp>::OpConversionPattern;

public:
  GetTupleElementOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::GetTupleElementOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::GetTupleElementOp getTupleElementOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto index = getTupleElementOp.index().getZExtValue();

    // TODO Consider adding template arguments to CallOp
    std::string calleStr =
        std::string("std::get<") + std::to_string(index) + std::string(">");
    StringAttr callee = rewriter.getStringAttr(calleStr);

    ArrayAttr args =
        rewriter.getArrayAttr({IntegerAttr::get(rewriter.getIndexType(), 0)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        getTupleElementOp, getTupleElementOp.getType(), callee, args, operands);

    return success();
  }
};

} // namespace

void populateMhloToEmitcPatterns(MLIRContext *ctx,
                                 OwningRewritePatternList &patterns) {
  /// Insert patterns for MHLO unary elementwise ops.
  patterns.insert<UnaryOpConversion<mhlo::AbsOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::abs");
  patterns.insert<UnaryOpConversion<mhlo::ConvertOp, emitc::CallOp>>(
      ctx, "mhlo::convert");
  patterns.insert<UnaryOpConversion<mhlo::CosOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::cos");
  patterns.insert<UnaryOpConversion<mhlo::ExpOp, emitc::CallOp>>(
      ctx, "mhlo::exponential");
  patterns.insert<UnaryOpConversion<mhlo::IsFiniteOp, emitc::CallOp>>(
      ctx, "mhlo::isfinite");
  patterns.insert<UnaryOpConversion<mhlo::LogOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::log");
  patterns.insert<UnaryOpConversion<mhlo::NegOp, emitc::CallOp>>(
      ctx, "mhlo::negate");
  patterns.insert<UnaryOpConversion<mhlo::SinOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::sin");
  patterns.insert<UnaryOpConversion<mhlo::SqrtOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::sqrt");

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

  // Insert patterns for MHLO MHLO binary logical elementwise ops.
  patterns.insert<BinaryOpConversion<mhlo::OrOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::or");
  patterns.insert<BinaryOpConversion<mhlo::XorOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::xor");

  // Insert patterns for MHLO tuple ops.
  // TODO:
  //  mhlo::CompareOp
  patterns.insert<TupleOpConversion>(ctx);
  patterns.insert<GetTupleElementOpConversion>(ctx);

  // Insert patterns for MHLO slice op.
  // TODO:
  //  mhlo::SliceOp
  //  mhlo::DynamicUpdateSliceOp

  // Insert patterns for other MHLO ops.
  // TODO:
  //  mhlo::BitcastConvertOp
  //  mhlo::BroadcastInDimOp
  patterns.insert<ConcatenateOpConversion>(ctx);

  // Insert patterns for MHLO RNG ops.
  // TODO:
  //  mhlo::RngUniformOp
  //  mhlo::RngBitGeneratorOp
}

namespace {

struct ConvertMhloToEmitcPass
    : public PassWrapper<ConvertMhloToEmitcPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect>();
  }
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<mhlo::AbsOp, mhlo::ConvertOp, mhlo::CosOp, mhlo::ExpOp,
                        mhlo::IsFiniteOp, mhlo::LogOp, mhlo::NegOp, mhlo::SinOp,
                        mhlo::SqrtOp>();
    target.addIllegalOp<mhlo::AddOp, mhlo::DivOp, mhlo::MaxOp, mhlo::MinOp,
                        mhlo::MulOp, mhlo::PowOp, mhlo::ShiftLeftOp,
                        mhlo::ShiftRightLogicalOp, mhlo::SubOp>();
    target.addIllegalOp<mhlo::OrOp, mhlo::XorOp>();
    target.addIllegalOp<mhlo::ConcatenateOp>();
    target.addIllegalOp<mhlo::TupleOp, mhlo::GetTupleElementOp>();

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
