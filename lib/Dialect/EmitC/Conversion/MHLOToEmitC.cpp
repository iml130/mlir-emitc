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

class BatchNormInferenceOpConversion
    : public OpConversionPattern<mhlo::BatchNormInferenceOp> {

public:
  BatchNormInferenceOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::BatchNormInferenceOp batchNormInferenceOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename mhlo::BatchNormInferenceOp::Adaptor adaptor(operands);

    StringRef funcName = "mhlo::batch_norm_inference";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 4> args_ = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int64_t>(0, operands.size()),
                        [&rewriter](int64_t i) -> Attribute {
                          return rewriter.getIndexAttr(i);
                        }));

    args_.push_back(batchNormInferenceOp.epsilonAttr());
    args_.push_back(batchNormInferenceOp.feature_indexAttr());

    ArrayAttr args = rewriter.getArrayAttr(args_);
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(batchNormInferenceOp.getResult().getType()),
         TypeAttr::get(adaptor.scale().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        batchNormInferenceOp, batchNormInferenceOp.getType(), callee, args,
        templateArgs, operands);

    return success();
  }
};

class BroadcastInDimOpConversion
    : public OpConversionPattern<mhlo::BroadcastInDimOp> {

public:
  BroadcastInDimOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::BroadcastInDimOp broadcastInDimOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "mhlo::broadcast_in_dim";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 4> args_ = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int64_t>(0, operands.size()),
                        [&rewriter](int64_t i) -> Attribute {
                          return rewriter.getIndexAttr(i);
                        }));

    args_.push_back(broadcastInDimOp.broadcast_dimensions());

    ArrayAttr args = rewriter.getArrayAttr(args_);

    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(broadcastInDimOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        broadcastInDimOp, broadcastInDimOp.getType(), callee, args,
        templateArgs, operands);

    return success();
  }
};

class ConcatenateOpConversion
    : public OpConversionPattern<mhlo::ConcatenateOp> {

public:
  ConcatenateOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::ConcatenateOp concatenateOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "mhlo::concatenate";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {rewriter.getI64IntegerAttr(concatenateOp.dimension()),
         TypeAttr::get(concatenateOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(concatenateOp,
                                               concatenateOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }
};

class ConvOpConversion : public OpConversionPattern<mhlo::ConvOp> {

public:
  ConvOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::ConvOp convOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename mhlo::ConvOp::Adaptor adaptor(operands);

    StringRef funcName = "mhlo::convolution";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 4> args_ = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int64_t>(0, operands.size()),
                        [&rewriter](int64_t i) -> Attribute {
                          return rewriter.getIndexAttr(i);
                        }));

    args_.push_back(convOp.batch_group_countAttr());
    args_.push_back(convOp.dimension_numbers().input_batch_dimension());
    args_.push_back(convOp.dimension_numbers().input_feature_dimension());
    args_.push_back(convOp.dimension_numbers().input_spatial_dimensions());
    args_.push_back(
        convOp.dimension_numbers().kernel_input_feature_dimension());
    args_.push_back(
        convOp.dimension_numbers().kernel_output_feature_dimension());
    args_.push_back(convOp.dimension_numbers().kernel_spatial_dimensions());
    args_.push_back(convOp.dimension_numbers().output_batch_dimension());
    args_.push_back(convOp.dimension_numbers().output_feature_dimension());
    args_.push_back(convOp.dimension_numbers().output_spatial_dimensions());
    args_.push_back(convOp.feature_group_countAttr());

    // Taken from mlir-hlo
    auto GetI64ElementsAttr = [&rewriter](ArrayRef<int64_t> values) {
      RankedTensorType ty = RankedTensorType::get(
          {static_cast<int64_t>(values.size())}, rewriter.getIntegerType(64));
      return DenseIntElementsAttr::get(ty, values);
    };

    args_.push_back(convOp.padding().getValueOr(GetI64ElementsAttr({0, 0})));
    args_.push_back(
        convOp.rhs_dilation().getValueOr(GetI64ElementsAttr({1, 1})));
    args_.push_back(
        convOp.window_strides().getValueOr(GetI64ElementsAttr({1, 1})));

    ArrayAttr args = rewriter.getArrayAttr(args_);
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(convOp.getResult().getType()),
                               TypeAttr::get(adaptor.lhs().getType()),
                               TypeAttr::get(adaptor.rhs().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(convOp, convOp.getType(), callee,
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

    StringRef comparisonDirection = compareOp.comparison_direction();
    Optional<StringRef> functionName =
        StringSwitch<Optional<StringRef>>(comparisonDirection)
            .Case("EQ", StringRef("std::equal_to"))
            .Case("NE", StringRef("std::not_equal_to"))
            .Case("GE", StringRef("std::greater_equal"))
            .Case("GT", StringRef("std::greater"))
            .Case("LE", StringRef("std::less_equal"))
            .Case("LT", StringRef("std::less"))
            .Default(None);

    if (!functionName.hasValue())
      return failure();

    Type elementType = compareOp.getOperand(0).getType();
    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(elementType),
         rewriter.getStringAttr(functionName.getValue())});

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
    StringRef funcName = "mhlo::slice";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 4> args_ = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int64_t>(0, operands.size()),
                        [&rewriter](int64_t i) -> Attribute {
                          return rewriter.getIndexAttr(i);
                        }));

    args_.push_back(sliceOp.start_indices());
    args_.push_back(sliceOp.limit_indices());
    args_.push_back(sliceOp.strides());

    ArrayAttr args = rewriter.getArrayAttr(args_);
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(sliceOp.getResult().getType())});

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
    StringRef funcName = "mhlo::dynamic_slice";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 4> args_ = llvm::to_vector<4>(
        llvm::map_range(llvm::seq<int64_t>(0, operands.size()),
                        [&rewriter](int64_t i) -> Attribute {
                          return rewriter.getIndexAttr(i);
                        }));

    args_.push_back(dynamicSliceOp.slice_sizes());

    ArrayAttr args = rewriter.getArrayAttr(args_);

    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(dynamicSliceOp.getResult().getType())});

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
    typename mhlo::DynamicUpdateSliceOp::Adaptor adaptor(operands);

    StringRef funcName = "mhlo::dynamic_update_slice";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(adaptor.update().getType())});

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
    StringRef funcName = "mhlo::rng_bit_generator";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(rngBitGeneratorOp.getResult().getType()),
         rewriter.getI32IntegerAttr(rngBitGeneratorOp.rng_algorithm())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        rngBitGeneratorOp, rngBitGeneratorOp.getType(), callee, args,
        templateArgs, operands);

    return success();
  }
};

} // namespace

void populateMhloToEmitcPatterns(MLIRContext *ctx,
                                 OwningRewritePatternList &patterns) {
  /// Insert patterns for MHLO unary elementwise ops.
  patterns.insert<CallOpConversion<mhlo::AbsOp>>(ctx, "mhlo::abs");
  patterns.insert<CallOpConversion<mhlo::CeilOp>>(ctx, "mhlo::ceil");
  patterns.insert<CallOpConversion<mhlo::ConvertOp>>(
      ctx, "mhlo::convert", /*explicitResultType=*/true);
  patterns.insert<CallOpConversion<mhlo::CosOp>>(ctx, "mhlo::cos");
  patterns.insert<CallOpConversion<mhlo::ExpOp>>(ctx, "mhlo::exponential");
  patterns.insert<CallOpConversion<mhlo::FloorOp>>(ctx, "mhlo::floor");
  patterns.insert<CallOpConversion<mhlo::IsFiniteOp>>(ctx, "mhlo::is_finite");
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
  patterns.insert<CallOpConversion<mhlo::OrOp>>(ctx, "mhlo::logical_or");
  patterns.insert<CallOpConversion<mhlo::XorOp>>(ctx, "mhlo::logical_xor");

  // Insert patterns for MHLO tuple ops.
  patterns.insert<CompareOpConversion>(ctx);
  patterns.insert<TupleOpConversion>(ctx);
  patterns.insert<GetTupleElementOpConversion>(ctx);

  // Insert patterns for MHLO slice ops.
  patterns.insert<SliceOpConversion>(ctx);
  patterns.insert<DynamicSliceOpConversion>(ctx);
  patterns.insert<DynamicUpdateSliceOpConversion>(ctx);

  // Insert patterns for other MHLO ops.
  patterns.insert<BatchNormInferenceOpConversion>(ctx);
  patterns.insert<BitcastConvertOpConversion>(ctx);
  patterns.insert<BroadcastInDimOpConversion>(ctx);
  patterns.insert<ConcatenateOpConversion>(ctx);
  patterns.insert<ConvOpConversion>(ctx);
  patterns.insert<CallOpConversion<mhlo::DotOp>>(ctx, "mhlo::dot",
                                                 /*explicitResultType=*/true);
  patterns.insert<CallOpConversion<mhlo::ReshapeOp>>(
      ctx, "mhlo::reshape", /*explicitResultType=*/true);
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
    registry.insert<emitc::EmitCDialect>();
  }
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    // clang-format off
    // MHLO unary elementwise ops
    target.addIllegalOp<mhlo::AbsOp,
                        mhlo::CeilOp,
                        mhlo::ConvertOp,
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
    target.addIllegalOp<mhlo::BatchNormInferenceOp,
                        mhlo::BitcastConvertOp,
                        mhlo::BroadcastInDimOp,
                        mhlo::ConcatenateOp,
                        mhlo::ConvOp,
                        mhlo::DotOp,
                        mhlo::ReshapeOp,
                        mhlo::SelectOp>();
    // MHLO RNG ops
    target.addIllegalOp<mhlo::RngUniformOp,
                        mhlo::RngBitGeneratorOp>();
    // clang-format on

    OwningRewritePatternList patterns;
    populateMhloToEmitcPatterns(&getContext(), patterns);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
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
