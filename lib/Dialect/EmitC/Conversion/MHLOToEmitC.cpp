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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitc {

namespace {

class BroadcastInDimOpConversion
    : public OpConversionPattern<mhlo::BroadcastInDimOp> {

public:
  BroadcastInDimOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::BroadcastInDimOp broadcastInDimOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = broadcastInDimOp.getLoc();
    auto broadcastDims = broadcastInDimOp.broadcast_dimensions();
    auto constOp = rewriter.create<ConstantOp>(loc, broadcastDims);

    std::vector<Value> newOperands(operands);
    newOperands.push_back(constOp.getResult());

    StringRef funcName = "mhlo::broadcast_in_dim";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(broadcastInDimOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        broadcastInDimOp, broadcastInDimOp.getType(), callee, args,
        templateArgs, newOperands);

    return success();
  }
};

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
    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(srcOp, srcOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }
};

template <typename SrcOp>
class CallOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  CallOpConversion(MLIRContext *ctx, StringRef funcName,
                   bool explicitResultType = false)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName),
        explicitResultType(explicitResultType) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename SrcOp::Adaptor srcAdapter(operands);

    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args;
    ArrayAttr templateArgs;

    if (explicitResultType) {
      Type resultType = srcOp.getType();
      templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(srcOp, srcOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }

  StringRef funcName;
  // If set, use the result type of the operation as the only template parameter
  bool explicitResultType;
};

class CompareOpConversion : public OpConversionPattern<mhlo::CompareOp> {
  using OpConversionPattern<mhlo::CompareOp>::OpConversionPattern;

public:
  CompareOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::CompareOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::CompareOp compareOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("mhlo::compare");

    llvm::StringRef comparisonDirection = compareOp.comparison_direction();
    llvm::StringRef functionName =
        llvm::StringSwitch<llvm::StringRef>(comparisonDirection)
            .Case("EQ", "std::equal_to")
            .Case("NE", "std::not_equal_to")
            .Case("GE", "std::greater_equal")
            .Case("GT", "std::greater")
            .Case("LE", "std::less_equal")
            .Case("LT", "std::less")
            .Default("");

    if (functionName.equals(""))
      return failure();

    Type elementType = compareOp.getOperand(0).getType();
    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(elementType), rewriter.getStringAttr(functionName)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        compareOp, compareOp.getType(), callee, args, templateArgs, operands);

    return success();
  }
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

    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        tupleOp, tupleOp.getType(), callee, args, templateArgs, operands);

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
    auto index = getTupleElementOp.index();

    // TODO: Consider adding template arguments to CallOp
    StringAttr callee = rewriter.getStringAttr("std::get");

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {IntegerAttr::get(rewriter.getIntegerType(32), index)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        getTupleElementOp, getTupleElementOp.getType(), callee, args,
        templateArgs, operands);

    return success();
  }
};

class SliceOpConversion : public OpConversionPattern<mhlo::SliceOp> {
  using OpConversionPattern<mhlo::SliceOp>::OpConversionPattern;

public:
  SliceOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::SliceOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::SliceOp sliceOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("mhlo::slice");

    auto operandTensorType =
        sliceOp.getOperand().getType().cast<RankedTensorType>();
    Type elementType = operandTensorType.getElementType();
    auto inputShape = operandTensorType.getShape();

    auto resultTensorType =
        sliceOp.getResult().getType().cast<RankedTensorType>();
    auto outputShape = resultTensorType.getShape();

    std::vector<Attribute> template_args_;
    template_args_.push_back(TypeAttr::get(elementType));
    for (auto index : sliceOp.start_indices().getIntValues()) {
      auto attr = rewriter.getI64IntegerAttr(index.getZExtValue());
      template_args_.push_back(attr);
    }
    for (auto index : sliceOp.limit_indices().getIntValues()) {
      auto attr = rewriter.getI64IntegerAttr(index.getZExtValue());
      template_args_.push_back(attr);
    }
    for (auto index : sliceOp.strides().getIntValues()) {
      auto attr = rewriter.getI64IntegerAttr(index.getZExtValue());
      template_args_.push_back(attr);
    }
    for (auto value : inputShape) {
      auto attr = rewriter.getI64IntegerAttr(value);
      template_args_.push_back(attr);
    }
    for (auto value : outputShape) {
      auto attr = rewriter.getI64IntegerAttr(value);
      template_args_.push_back(attr);
    }

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(template_args_);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        sliceOp, sliceOp.getType(), callee, args, templateArgs, operands);

    return success();
  }
};

class DynamicSliceOpConversion
    : public OpConversionPattern<mhlo::DynamicSliceOp> {
  using OpConversionPattern<mhlo::DynamicSliceOp>::OpConversionPattern;

public:
  DynamicSliceOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::DynamicSliceOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::DynamicSliceOp dynamicSliceOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("mhlo::dynamic_slice");

    auto operandTensorType =
        dynamicSliceOp.getOperand(0).getType().cast<RankedTensorType>();
    Type elementType = operandTensorType.getElementType();
    auto inputShape = operandTensorType.getShape();

    auto resultTensorType =
        dynamicSliceOp.getResult().getType().cast<RankedTensorType>();
    auto outputShape = resultTensorType.getShape();

    std::vector<Attribute> template_args_;
    template_args_.push_back(TypeAttr::get(elementType));
    for (auto size : dynamicSliceOp.slice_sizes().getIntValues()) {
      auto attr = rewriter.getI64IntegerAttr(size.getZExtValue());
      template_args_.push_back(attr);
    }
    for (auto value : inputShape) {
      auto attr = rewriter.getI64IntegerAttr(value);
      template_args_.push_back(attr);
    }
    for (auto value : outputShape) {
      auto attr = rewriter.getI64IntegerAttr(value);
      template_args_.push_back(attr);
    }

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(template_args_);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(dynamicSliceOp,
                                               dynamicSliceOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }
};

class DynamicUpdateSliceOpConversion
    : public OpConversionPattern<mhlo::DynamicUpdateSliceOp> {
  using OpConversionPattern<mhlo::DynamicUpdateSliceOp>::OpConversionPattern;

public:
  DynamicUpdateSliceOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::DynamicUpdateSliceOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::DynamicUpdateSliceOp dynamicUpdateSliceOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("mhlo::dynamic_update_slice");

    auto operandTensorType =
        dynamicUpdateSliceOp.getOperand(0).getType().cast<RankedTensorType>();
    Type elementType = operandTensorType.getElementType();
    auto inputShape = operandTensorType.getShape();

    auto updateTensorType =
        dynamicUpdateSliceOp.getOperand(1).getType().cast<RankedTensorType>();
    auto updateShape = updateTensorType.getShape();

    std::vector<Attribute> template_args_;
    template_args_.push_back(TypeAttr::get(elementType));
    for (auto value : inputShape) {
      auto attr = rewriter.getI64IntegerAttr(value);
      template_args_.push_back(attr);
    }
    for (auto value : updateShape) {
      auto attr = rewriter.getI64IntegerAttr(value);
      template_args_.push_back(attr);
    }

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(template_args_);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        dynamicUpdateSliceOp, dynamicUpdateSliceOp.getType(), callee, args,
        templateArgs, operands);

    return success();
  }
};

class BitcastConvertOpConversion
    : public OpConversionPattern<mhlo::BitcastConvertOp> {
  using OpConversionPattern<mhlo::BitcastConvertOp>::OpConversionPattern;

public:
  BitcastConvertOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::BitcastConvertOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::BitcastConvertOp bitcastConvertOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("mhlo::bitcast_convert");

    ArrayAttr args;
    Type resultType = bitcastConvertOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        bitcastConvertOp, bitcastConvertOp.getType(), callee, args,
        templateArgs, operands);

    return success();
  }
};

class ConvertOpConversion : public OpConversionPattern<mhlo::ConvertOp> {
  using OpConversionPattern<mhlo::ConvertOp>::OpConversionPattern;

public:
  ConvertOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::ConvertOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::ConvertOp convertOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("mhlo::convert");

    Type elementType = convertOp.getType();
    if (auto tensorType = elementType.dyn_cast<TensorType>()) {
      elementType = tensorType.getElementType();
    }
    ArrayAttr args;
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(elementType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        convertOp, convertOp.getType(), callee, args, templateArgs, operands);

    return success();
  }
};

class ReshapeOpConversion : public OpConversionPattern<mhlo::ReshapeOp> {
  using OpConversionPattern<mhlo::ReshapeOp>::OpConversionPattern;

public:
  ReshapeOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::ReshapeOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::ReshapeOp reshapeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("mhlo::reshape");

    // We might need to add arguments if tensor rank/shape get modelled in the
    // translation
    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        reshapeOp, reshapeOp.getType(), callee, args, templateArgs, operands);

    return success();
  }
};

class SelectOpConversion : public OpConversionPattern<mhlo::SelectOp> {
  using OpConversionPattern<mhlo::SelectOp>::OpConversionPattern;

public:
  SelectOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::SelectOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::SelectOp selectOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("mhlo::select");
    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        selectOp, selectOp.getType(), callee, args, templateArgs, operands);

    return success();
  }
};

class RngBitGeneratorOpConversion
    : public OpConversionPattern<mhlo::RngBitGeneratorOp> {
  using OpConversionPattern<mhlo::RngBitGeneratorOp>::OpConversionPattern;

public:
  RngBitGeneratorOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::RngBitGeneratorOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::RngBitGeneratorOp rngBitGeneratorOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename mhlo::RngUniformOp::Adaptor srcAdapter(operands);

    StringAttr callee = rewriter.getStringAttr("mhlo::rng_bit_generator");

    if (auto tupleType = rngBitGeneratorOp.getType().dyn_cast<TupleType>()) {
      if (tupleType.getTypes().size() == 2) {
        if (auto tensorType =
                tupleType.getTypes()[1].dyn_cast<RankedTensorType>()) {

          Type elementType = tensorType.getElementType();
          int64_t size = tensorType.getNumElements();

          ArrayAttr args;
          ArrayAttr templateArgs =
              rewriter.getArrayAttr({TypeAttr::get(elementType),
                                     rngBitGeneratorOp.rng_algorithmAttr(),
                                     rewriter.getI64IntegerAttr(size)});

          rewriter.replaceOpWithNewOp<emitc::CallOp>(
              rngBitGeneratorOp, rngBitGeneratorOp.getType(), callee, args,
              templateArgs, operands);

          return success();
        }
      }
    }

    return failure();
  }
};

} // namespace

void populateMhloToEmitcPatterns(MLIRContext *ctx,
                                 OwningRewritePatternList &patterns) {
  /// Insert patterns for MHLO unary elementwise ops.
  patterns.insert<CallOpConversion<mhlo::AbsOp>>(ctx, "mhlo::abs");
  patterns.insert<CallOpConversion<mhlo::CeilOp>>(ctx, "mhlo::ceil");
  patterns.insert<CallOpConversion<mhlo::CosOp>>(ctx, "mhlo::cos");
  patterns.insert<CallOpConversion<mhlo::ExpOp>>(ctx, "mhlo::exponential");
  patterns.insert<CallOpConversion<mhlo::FloorOp>>(ctx, "mhlo::floor");
  patterns.insert<CallOpConversion<mhlo::IsFiniteOp>>(ctx, "mhlo::isfinite");
  patterns.insert<CallOpConversion<mhlo::LogOp>>(ctx, "mhlo::log");
  patterns.insert<CallOpConversion<mhlo::NegOp>>(ctx, "mhlo::negate");
  patterns.insert<CallOpConversion<mhlo::RoundOp>>(ctx, "mhlo::round");
  patterns.insert<CallOpConversion<mhlo::SinOp>>(ctx, "mhlo::sin");
  patterns.insert<CallOpConversion<mhlo::SqrtOp>>(ctx, "mhlo::sqrt");
  patterns.insert<CallOpConversion<mhlo::TanhOp>>(ctx, "mhlo::tanh");

  /// Insert patterns for MHLO binary elementwise ops.
  patterns.insert<CallOpConversion<mhlo::AddOp>>(ctx, "mhlo::add");
  patterns.insert<CallOpConversion<mhlo::Atan2Op>>(ctx, "mhlo::atan2");
  patterns.insert<CallOpConversion<mhlo::DivOp>>(ctx, "mhlo::div");
  patterns.insert<CallOpConversion<mhlo::MaxOp>>(ctx, "mhlo::max");
  patterns.insert<CallOpConversion<mhlo::MinOp>>(ctx, "mhlo::min");
  patterns.insert<CallOpConversion<mhlo::MulOp>>(ctx, "mhlo::mul");
  patterns.insert<CallOpConversion<mhlo::PowOp>>(ctx, "mhlo::pow");
  patterns.insert<CallOpConversion<mhlo::ShiftLeftOp>>(ctx, "mhlo::shift_left");
  patterns.insert<CallOpConversion<mhlo::ShiftRightLogicalOp>>(
      ctx, "mhlo::shift_right_logical");
  patterns.insert<CallOpConversion<mhlo::SubOp>>(ctx, "mhlo::sub");

  // Insert patterns for MHLO MHLO binary logical elementwise ops.
  patterns.insert<CallOpConversion<mhlo::OrOp>>(ctx, "mhlo::or");
  patterns.insert<CallOpConversion<mhlo::XorOp>>(ctx, "mhlo::xor");

  // Insert patterns for MHLO tuple ops.
  patterns.insert<CompareOpConversion>(ctx);
  patterns.insert<TupleOpConversion>(ctx);
  patterns.insert<GetTupleElementOpConversion>(ctx);

  // Insert patterns for MHLO slice ops.
  patterns.insert<SliceOpConversion>(ctx);
  patterns.insert<DynamicSliceOpConversion>(ctx);
  patterns.insert<DynamicUpdateSliceOpConversion>(ctx);

  // Insert patterns for other MHLO ops.
  patterns.insert<BitcastConvertOpConversion>(ctx);
  patterns.insert<BroadcastInDimOpConversion>(ctx);
  patterns.insert<ConvertOpConversion>(ctx);
  patterns.insert<ConcatenateOpConversion>(ctx);
  patterns.insert<ReshapeOpConversion>(ctx);
  patterns.insert<SelectOpConversion>(ctx);

  // Insert patterns for MHLO RNG ops.
  patterns.insert<CallOpConversion<mhlo::RngUniformOp>>(
      ctx, "mhlo::rng_uniform", /*explicitResultType=*/true);
  patterns.insert<RngBitGeneratorOpConversion>(ctx);
}

namespace {

struct ConvertMhloToEmitcPass
    : public PassWrapper<ConvertMhloToEmitcPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, mlir::StandardOpsDialect>();
  }
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    target.addLegalOp<ConstantOp>();
    // clang-format off
    // MHLO unary elementwise ops
    target.addIllegalOp<mhlo::AbsOp,
                        mhlo::CeilOp,
                        mhlo::CosOp,
                        mhlo::ExpOp,
                        mhlo::FloorOp,
                        mhlo::IsFiniteOp,
                        mhlo::LogOp,
                        mhlo::NegOp,
                        mhlo::RoundOp,
                        mhlo::SinOp,
                        mhlo::SqrtOp,
                        mhlo::TanhOp>();
    // MHLO binary elementwise ops
    target.addIllegalOp<mhlo::AddOp,
                        mhlo::Atan2Op,
                        mhlo::DivOp,
                        mhlo::MaxOp,
                        mhlo::MinOp,
                        mhlo::MulOp,
                        mhlo::PowOp,
                        mhlo::ShiftLeftOp,
                        mhlo::ShiftRightLogicalOp,
                        mhlo::SubOp>();
    // MHLO binary logical elementwise ops
    target.addIllegalOp<mhlo::OrOp,
                        mhlo::XorOp>();
    // MHLO tuple ops
    target.addIllegalOp<mhlo::CompareOp,
                        mhlo::TupleOp,
                        mhlo::GetTupleElementOp>();
    // MHLO slice ops
    target.addIllegalOp<mhlo::DynamicSliceOp,
                        mhlo::DynamicUpdateSliceOp,
                        mhlo::SliceOp>();
    // other MHLO ops
    target.addIllegalOp<mhlo::BitcastConvertOp,
                        mhlo::BroadcastInDimOp,
                        mhlo::ConvertOp,
                        mhlo::ConcatenateOp,
                        mhlo::ReshapeOp,
                        mhlo::SelectOp>();
    // MHLO RNG ops
    target.addIllegalOp<mhlo::RngUniformOp,
                        mhlo::RngBitGeneratorOp>();
    // clang-format on

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
