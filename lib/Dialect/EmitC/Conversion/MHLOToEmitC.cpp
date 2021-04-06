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

#include "PassDetail.h"
#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "emitc/Dialect/EmitC/Passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace emitc {

namespace {

/// Common functions.
/// Adopted from mlir-hlo.
DenseIntElementsAttr i64ElementsAttr(int64_t value, size_t count,
                                     MLIRContext *ctx) {
  RankedTensorType ty = RankedTensorType::get({static_cast<int64_t>(count)},
                                              IntegerType::get(ctx, 64));
  SmallVector<int64_t, 4> values(count, value);
  return DenseIntElementsAttr::get(ty, values);
}

SmallVector<Attribute, 2> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::to_vector<2>(
      llvm::map_range(llvm::seq<int64_t>(0, n), [&ctx](int64_t i) -> Attribute {
        return IntegerAttr::get(IndexType::get(ctx), i);
      }));
}

/// Convert `mhlo.constant` into an `emitc.const` operation.
class ConstOpConversion : public OpRewritePattern<mhlo::ConstOp> {
public:
  using OpRewritePattern<mhlo::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConstOp constOp,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstOp>(constOp, constOp.getType(),
                                                constOp.value());
    return success();
  }
};

/// Convert `mhlo.batch_norm_inference` into an `emitc.const` operation.
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

    SmallVector<Attribute, 2> args_ =
        indexSequence(operands.size(), batchNormInferenceOp.getContext());

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

/// Convert `mhlo.broadcast_in_dim` into an `emitc.call` operation.
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

    SmallVector<Attribute, 2> args_ =
        indexSequence(operands.size(), broadcastInDimOp.getContext());

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

/// Convert `mhlo.concatenate` into an `emitc.call` operation.
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

/// Convert `mhlo.convolution` into an `emitc.call` operation.
class ConvOpConversion : public OpConversionPattern<mhlo::ConvOp> {

public:
  ConvOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::ConvOp convOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename mhlo::ConvOp::Adaptor adaptor(operands);
    auto ctx = convOp.getContext();

    StringRef funcName = "mhlo::convolution";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> args_ =
        indexSequence(operands.size(), convOp.getContext());

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

    args_.push_back(convOp.padding().getValueOr(i64ElementsAttr(0, 2, ctx)));
    args_.push_back(
        convOp.lhs_dilation().getValueOr(i64ElementsAttr(1, 2, ctx)));
    args_.push_back(
        convOp.rhs_dilation().getValueOr(i64ElementsAttr(1, 2, ctx)));
    args_.push_back(
        convOp.window_strides().getValueOr(i64ElementsAttr(1, 2, ctx)));

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

/// Convert a common `mhlo` operation into an `emitc.call` operation.
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
    ArrayAttr templateArgs;
    if (!templateArgs_.empty()) {
      templateArgs = ArrayAttr::get(srcOp.getContext(), templateArgs_);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(srcOp, srcOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }

  StringRef funcName;
  // If set, use the result type of the operation as template parameter.
  bool explicitResultType;
  // If set, use the operand types as (additional) template parameters.
  bool explicitOperandTypes;
};

/// Convert `mhlo.compare` into an `emitc.call` operation.
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

/// Convert `mhlo.get_tuple_element` into an `emitc.call` operation.
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

/// Convert `mhlo.slice` into an `emitc.call` operation.
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

    SmallVector<Attribute, 2> args_ =
        indexSequence(operands.size(), sliceOp.getContext());

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

/// Convert `mhlo.dynamic-slice` into an `emitc.call` operation.
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

    SmallVector<Attribute, 2> args_ =
        indexSequence(operands.size(), dynamicSliceOp.getContext());

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

/// Convert `mhlo.dynamic-update-slice` into an `emitc.call` operation.
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

/// Convert `mhlo.pad` into an `emitc.call` operation.
class PadOpConversion : public OpConversionPattern<mhlo::PadOp> {
  using OpConversionPattern<mhlo::PadOp>::OpConversionPattern;

public:
  PadOpConversion(MLIRContext *ctx) : OpConversionPattern<mhlo::PadOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::PadOp padOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("mhlo::pad");

    SmallVector<Attribute, 2> args_ =
        indexSequence(operands.size(), padOp.getContext());

    args_.push_back(padOp.edge_padding_low());
    args_.push_back(padOp.edge_padding_high());
    args_.push_back(padOp.interior_padding());

    ArrayAttr args = rewriter.getArrayAttr(args_);

    Type resultType = padOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(padOp, padOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }
};

/// Convert `mhlo.rng_bit_generator` into an `emitc.call` operation.
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
  // Insert patterns for MHLO nullary ops.
  patterns.insert<ConstOpConversion>(ctx);

  // Insert patterns for MHLO unary elementwise ops.
  patterns.insert<CallOpConversion<mhlo::AbsOp>>(ctx, "mhlo::abs");
  patterns.insert<CallOpConversion<mhlo::CeilOp>>(ctx, "mhlo::ceil");
  patterns.insert<CallOpConversion<mhlo::ConvertOp>>(
      ctx, "mhlo::convert", /*explicitResultType=*/true);
  patterns.insert<CallOpConversion<mhlo::CosOp>>(ctx, "mhlo::cos");
  patterns.insert<CallOpConversion<mhlo::ExpOp>>(ctx, "mhlo::exponential");
  patterns.insert<CallOpConversion<mhlo::Expm1Op>>(
      ctx, "mhlo::exponential_minus_one");
  patterns.insert<CallOpConversion<mhlo::FloorOp>>(ctx, "mhlo::floor");
  patterns.insert<CallOpConversion<mhlo::IsFiniteOp>>(ctx, "mhlo::is_finite");
  patterns.insert<CallOpConversion<mhlo::LogOp>>(ctx, "mhlo::log");
  patterns.insert<CallOpConversion<mhlo::Log1pOp>>(ctx, "mhlo::log_plus_one");
  patterns.insert<CallOpConversion<mhlo::NegOp>>(ctx, "mhlo::negate");
  patterns.insert<CallOpConversion<mhlo::RoundOp>>(ctx, "mhlo::round");
  patterns.insert<CallOpConversion<mhlo::SinOp>>(ctx, "mhlo::sin");
  patterns.insert<CallOpConversion<mhlo::SqrtOp>>(ctx, "mhlo::sqrt");
  patterns.insert<CallOpConversion<mhlo::TanhOp>>(ctx, "mhlo::tanh");

  // Insert patterns for MHLO binary elementwise ops.
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

  // Insert patterns for MHLO binary logical elementwise ops.
  patterns.insert<CallOpConversion<mhlo::OrOp>>(ctx, "mhlo::logical_or");
  patterns.insert<CallOpConversion<mhlo::XorOp>>(ctx, "mhlo::logical_xor");

  // Insert patterns for MHLO tuple ops.
  patterns.insert<CompareOpConversion>(ctx);
  patterns.insert<CallOpConversion<mhlo::TupleOp>>(ctx, "std::make_tuple");
  patterns.insert<GetTupleElementOpConversion>(ctx);

  // Insert patterns for MHLO slice ops.
  patterns.insert<SliceOpConversion>(ctx);
  patterns.insert<DynamicSliceOpConversion>(ctx);
  patterns.insert<DynamicUpdateSliceOpConversion>(ctx);

  // Insert patterns for other MHLO ops.
  patterns.insert<BatchNormInferenceOpConversion>(ctx);
  patterns.insert<CallOpConversion<mhlo::BitcastConvertOp>>(
      ctx, "mhlo::bitcast_convert", /*explicitResultType=*/true);
  patterns.insert<BroadcastInDimOpConversion>(ctx);
  patterns.insert<CallOpConversion<mhlo::ClampOp>>(
      ctx, "mhlo::clamp", /*explicitResultType=*/false,
      /*explicitOperandTypes=*/true);
  patterns.insert<ConcatenateOpConversion>(ctx);
  patterns.insert<ConvOpConversion>(ctx);
  patterns.insert<CallOpConversion<mhlo::DotOp>>(ctx, "mhlo::dot",
                                                 /*explicitResultType=*/true);
  patterns.insert<PadOpConversion>(ctx);
  patterns.insert<CallOpConversion<mhlo::ReshapeOp>>(
      ctx, "mhlo::reshape", /*explicitResultType=*/true);
  patterns.insert<CallOpConversion<mhlo::SelectOp>>(ctx, "mhlo::select");

  // Insert patterns for MHLO RNG ops.
  patterns.insert<CallOpConversion<mhlo::RngUniformOp>>(
      ctx, "mhlo::rng_uniform", /*explicitResultType=*/true);
  patterns.insert<RngBitGeneratorOpConversion>(ctx);
}

namespace {

struct ConvertMhloToEmitCPass
    : public ConvertMHLOToEmitCBase<ConvertMhloToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<FuncOp>();
    target.addLegalOp<ModuleOp>();

    // clang-format off
    // MHLO nullary ops
    target.addIllegalOp<mhlo::ConstOp>();
    // MHLO unary elementwise ops
    target.addIllegalOp<mhlo::AbsOp,
                        mhlo::CeilOp,
                        mhlo::ConvertOp,
                        mhlo::CosOp,
                        mhlo::ExpOp,
                        mhlo::Expm1Op,
                        mhlo::FloorOp,
                        mhlo::IsFiniteOp,
                        mhlo::LogOp,
                        mhlo::Log1pOp,
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
    // MHLO region ops
    target.addIllegalOp<mhlo::ReduceOp,
                        mhlo::ReturnOp>();
    // Other MHLO ops
    target.addIllegalOp<mhlo::BatchNormInferenceOp,
                        mhlo::BitcastConvertOp,
                        mhlo::BroadcastInDimOp,
                        mhlo::ClampOp,
                        mhlo::ConcatenateOp,
                        mhlo::ConvOp,
                        mhlo::DotOp,
                        mhlo::PadOp,
                        mhlo::ReshapeOp,
                        mhlo::SelectOp>();
    // MHLO RNG ops
    target.addIllegalOp<mhlo::RngUniformOp,
                        mhlo::RngBitGeneratorOp>();
    // clang-format on

    OwningRewritePatternList patterns(&getContext());
    populateMhloToEmitcPatterns(&getContext(), patterns);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<FunctionPass> createConvertMhloToEmitCPass() {
  return std::make_unique<ConvertMhloToEmitCPass>();
}

} // namespace emitc
} // namespace mlir
