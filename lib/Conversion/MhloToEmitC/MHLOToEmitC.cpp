//===- MHLOToEmitC.cpp - MHLO to EmitC conversion -------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for lowering MHLO dialect to EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"
#include "emitc/Conversion/MhloToEmitC/MhloToEmitC.h"

using namespace mlir;
using namespace mlir::emitc;

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

/// Convert `mhlo.constant` into an `emitc.constant` operation.
class ConstOpConversion : public OpRewritePattern<mhlo::ConstantOp> {
public:
  using OpRewritePattern<mhlo::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConstantOp constOp,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, constOp.getType(),
                                                   constOp.getValue());
    return success();
  }
};

/// Convert `mhlo.batch_norm_inference` into an `emitc.call` operation.
class BatchNormInferenceOpConversion
    : public OpConversionPattern<mhlo::BatchNormInferenceOp> {

public:
  BatchNormInferenceOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::BatchNormInferenceOp batchNormInferenceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "emitc::mhlo::batch_norm_inference";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments = indexSequence(
        adaptor.getOperands().size(), batchNormInferenceOp.getContext());

    arguments.push_back(batchNormInferenceOp.getEpsilonAttr());
    arguments.push_back(batchNormInferenceOp.getFeatureIndexAttr());

    ArrayAttr args = rewriter.getArrayAttr(arguments);
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(batchNormInferenceOp.getResult().getType()),
         TypeAttr::get(adaptor.getScale().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        batchNormInferenceOp, batchNormInferenceOp.getType(), callee, args,
        templateArgs, adaptor.getOperands());

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
  matchAndRewrite(mhlo::BroadcastInDimOp broadcastInDimOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::mhlo::broadcast_in_dim";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments = indexSequence(
        adaptor.getOperands().size(), broadcastInDimOp.getContext());

    arguments.push_back(broadcastInDimOp.getBroadcastDimensions());

    ArrayAttr args = rewriter.getArrayAttr(arguments);

    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(broadcastInDimOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        broadcastInDimOp, broadcastInDimOp.getType(), callee, args,
        templateArgs, adaptor.getOperands());

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
  matchAndRewrite(mhlo::ConcatenateOp concatenateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "emitc::mhlo::concatenate";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {rewriter.getI64IntegerAttr(concatenateOp.getDimension()),
         TypeAttr::get(concatenateOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        concatenateOp, concatenateOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};

/// Convert `mhlo.convolution` into an `emitc.call` operation.
class ConvOpConversion : public OpConversionPattern<mhlo::ConvolutionOp> {

public:
  ConvOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::ConvolutionOp convOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = convOp.getContext();

    StringRef funcName = "emitc::mhlo::convolution";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments =
        indexSequence(adaptor.getOperands().size(), convOp.getContext());

    arguments.push_back(convOp.getBatchGroupCountAttr());
    arguments.push_back(rewriter.getI64IntegerAttr(
        convOp.getDimensionNumbers().getInputBatchDimension()));
    arguments.push_back(rewriter.getI64IntegerAttr(
        convOp.getDimensionNumbers().getInputFeatureDimension()));
    arguments.push_back(rewriter.getI64TensorAttr(
        convOp.getDimensionNumbers().getInputSpatialDimensions()));
    arguments.push_back(rewriter.getI64IntegerAttr(
        convOp.getDimensionNumbers().getKernelInputFeatureDimension()));
    arguments.push_back(rewriter.getI64IntegerAttr(
        convOp.getDimensionNumbers().getKernelOutputFeatureDimension()));
    arguments.push_back(rewriter.getI64TensorAttr(
        convOp.getDimensionNumbers().getKernelSpatialDimensions()));
    arguments.push_back(rewriter.getI64IntegerAttr(
        convOp.getDimensionNumbers().getOutputBatchDimension()));
    arguments.push_back(rewriter.getI64IntegerAttr(
        convOp.getDimensionNumbers().getOutputFeatureDimension()));
    arguments.push_back(rewriter.getI64TensorAttr(
        convOp.getDimensionNumbers().getOutputSpatialDimensions()));
    arguments.push_back(convOp.getFeatureGroupCountAttr());

    arguments.push_back(
        convOp.getPadding().value_or(i64ElementsAttr(0, 2, ctx)));
    arguments.push_back(
        convOp.getLhsDilation().value_or(i64ElementsAttr(1, 2, ctx)));
    arguments.push_back(
        convOp.getRhsDilation().value_or(i64ElementsAttr(1, 2, ctx)));
    arguments.push_back(
        convOp.getWindowStrides().value_or(i64ElementsAttr(1, 2, ctx)));

    ArrayAttr args = rewriter.getArrayAttr(arguments);
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(convOp.getResult().getType()),
                               TypeAttr::get(adaptor.getLhs().getType()),
                               TypeAttr::get(adaptor.getRhs().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(convOp, convOp.getType(), callee,
                                               args, templateArgs,
                                               adaptor.getOperands());

    return success();
  }
};

/// Convert a common `mhlo` operation into an `emitc.call` operation.
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

/// Convert `mhlo.compare` into an `emitc.call` operation.
class CompareOpConversion : public OpConversionPattern<mhlo::CompareOp> {
  using OpConversionPattern<mhlo::CompareOp>::OpConversionPattern;

public:
  CompareOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::CompareOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::CompareOp compareOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = compareOp.getContext();

    StringAttr callee = rewriter.getStringAttr("emitc::mhlo::compare");

    mhlo::ComparisonDirection comparisonDirection =
        compareOp.getComparisonDirection();
    Optional<StringRef> functionName =
        StringSwitch<Optional<StringRef>>(
            stringifyComparisonDirection(comparisonDirection))
            .Case("EQ", StringRef("std::equal_to"))
            .Case("NE", StringRef("std::not_equal_to"))
            .Case("GE", StringRef("std::greater_equal"))
            .Case("GT", StringRef("std::greater"))
            .Case("LE", StringRef("std::less_equal"))
            .Case("LT", StringRef("std::less"))
            .Default(None);

    if (!functionName.has_value())
      return failure();

    Type elementType = compareOp.getOperand(0).getType();
    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(elementType),
         emitc::OpaqueAttr::get(ctx, functionName.value())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(compareOp, compareOp.getType(),
                                               callee, args, templateArgs,
                                               adaptor.getOperands());

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
  matchAndRewrite(mhlo::GetTupleElementOp getTupleElementOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto index = getTupleElementOp.getIndex();

    StringAttr callee = rewriter.getStringAttr("std::get");

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {IntegerAttr::get(rewriter.getIntegerType(32), index)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        getTupleElementOp, getTupleElementOp.getType(), callee, args,
        templateArgs, adaptor.getOperands());

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
  matchAndRewrite(mhlo::SliceOp sliceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::mhlo::slice";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments =
        indexSequence(adaptor.getOperands().size(), sliceOp.getContext());

    arguments.push_back(sliceOp.getStartIndices());
    arguments.push_back(sliceOp.getLimitIndices());
    arguments.push_back(sliceOp.getStrides());

    ArrayAttr args = rewriter.getArrayAttr(arguments);
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(sliceOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(sliceOp, sliceOp.getType(),
                                               callee, args, templateArgs,
                                               adaptor.getOperands());

    return success();
  }
};

/// Convert `mhlo.dynamic_slice` into an `emitc.call` operation.
class DynamicSliceOpConversion
    : public OpConversionPattern<mhlo::DynamicSliceOp> {
  using OpConversionPattern<mhlo::DynamicSliceOp>::OpConversionPattern;

public:
  DynamicSliceOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::DynamicSliceOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::DynamicSliceOp dynamicSliceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::mhlo::dynamic_slice";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments = indexSequence(
        adaptor.getOperands().size(), dynamicSliceOp.getContext());

    arguments.push_back(dynamicSliceOp.getSliceSizes());

    ArrayAttr args = rewriter.getArrayAttr(arguments);

    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(dynamicSliceOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        dynamicSliceOp, dynamicSliceOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};

/// Convert `mhlo.dynamic_update_slice` into an `emitc.call` operation.
class DynamicUpdateSliceOpConversion
    : public OpConversionPattern<mhlo::DynamicUpdateSliceOp> {
  using OpConversionPattern<mhlo::DynamicUpdateSliceOp>::OpConversionPattern;

public:
  DynamicUpdateSliceOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::DynamicUpdateSliceOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::DynamicUpdateSliceOp dynamicUpdateSliceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "emitc::mhlo::dynamic_update_slice";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(adaptor.getUpdate().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        dynamicUpdateSliceOp, dynamicUpdateSliceOp.getType(), callee, args,
        templateArgs, adaptor.getOperands());

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
  matchAndRewrite(mhlo::PadOp padOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::mhlo::pad");

    SmallVector<Attribute, 2> arguments =
        indexSequence(adaptor.getOperands().size(), padOp.getContext());

    arguments.push_back(padOp.getEdgePaddingLow());
    arguments.push_back(padOp.getEdgePaddingHigh());
    arguments.push_back(padOp.getInteriorPadding());

    ArrayAttr args = rewriter.getArrayAttr(arguments);

    Type resultType = padOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(padOp, padOp.getType(), callee,
                                               args, templateArgs,
                                               adaptor.getOperands());

    return success();
  }
};

/// Convert `mhlo.rng` into an `emitc.call` operation.
class RngOpConversion : public OpConversionPattern<mhlo::RngOp> {

public:
  RngOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::RngOp rngOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (rngOp.getRngDistribution() != mhlo::RngDistribution::UNIFORM) {
      return rngOp.emitError(
          "Distributions other than uniform are not supported.");
    }

    StringRef funcName = "emitc::mhlo::rng_uniform";
    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args;

    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(rngOp.getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(rngOp, rngOp.getType(), callee,
                                               args, templateArgs,
                                               adaptor.getOperands());

    return success();
  }
};

} // namespace

void populateMhloToEmitcPatterns(MLIRContext *ctx,
                                 RewritePatternSet &patterns) {
  // Insert patterns for MHLO nullary ops.
  patterns.add<ConstOpConversion>(ctx);

  // Insert patterns for MHLO unary elementwise ops.
  patterns.add<CallOpConversion<mhlo::AbsOp>>(ctx, "emitc::mhlo::abs");
  patterns.add<CallOpConversion<mhlo::CeilOp>>(ctx, "emitc::mhlo::ceil");
  patterns.add<CallOpConversion<mhlo::ConvertOp>>(ctx, "emitc::mhlo::convert",
                                                  /*explicitResultType=*/true);
  patterns.add<CallOpConversion<mhlo::CosineOp>>(ctx, "emitc::mhlo::cos");
  patterns.add<CallOpConversion<mhlo::ExpOp>>(ctx, "emitc::mhlo::exponential");
  patterns.add<CallOpConversion<mhlo::Expm1Op>>(
      ctx, "emitc::mhlo::exponential_minus_one");
  patterns.add<CallOpConversion<mhlo::FloorOp>>(ctx, "emitc::mhlo::floor");
  patterns.add<CallOpConversion<mhlo::IsFiniteOp>>(ctx,
                                                   "emitc::mhlo::is_finite");
  patterns.add<CallOpConversion<mhlo::LogOp>>(ctx, "emitc::mhlo::log");
  patterns.add<CallOpConversion<mhlo::Log1pOp>>(ctx,
                                                "emitc::mhlo::log_plus_one");
  patterns.add<CallOpConversion<mhlo::NegOp>>(ctx, "emitc::mhlo::negate");
  patterns.add<CallOpConversion<mhlo::RoundOp>>(ctx, "emitc::mhlo::round");
  patterns.add<CallOpConversion<mhlo::SineOp>>(ctx, "emitc::mhlo::sin");
  patterns.add<CallOpConversion<mhlo::SqrtOp>>(ctx, "emitc::mhlo::sqrt");
  patterns.add<CallOpConversion<mhlo::TanhOp>>(ctx, "emitc::mhlo::tanh");

  // Insert patterns for MHLO binary elementwise ops.
  patterns.add<CallOpConversion<mhlo::AddOp>>(ctx, "emitc::mhlo::add");
  patterns.add<CallOpConversion<mhlo::Atan2Op>>(ctx, "emitc::mhlo::atan2");
  patterns.add<CallOpConversion<mhlo::DivOp>>(ctx, "emitc::mhlo::div");
  patterns.add<CallOpConversion<mhlo::MaxOp>>(ctx, "emitc::mhlo::max");
  patterns.add<CallOpConversion<mhlo::MinOp>>(ctx, "emitc::mhlo::min");
  patterns.add<CallOpConversion<mhlo::MulOp>>(ctx, "emitc::mhlo::mul");
  patterns.add<CallOpConversion<mhlo::PowOp>>(ctx, "emitc::mhlo::pow");
  patterns.add<CallOpConversion<mhlo::ShiftLeftOp>>(ctx,
                                                    "emitc::mhlo::shift_left");
  patterns.add<CallOpConversion<mhlo::ShiftRightLogicalOp>>(
      ctx, "emitc::mhlo::shift_right_logical");
  patterns.add<CallOpConversion<mhlo::SubtractOp>>(ctx, "emitc::mhlo::sub");

  // Insert patterns for MHLO binary logical elementwise ops.
  patterns.add<CallOpConversion<mhlo::OrOp>>(ctx, "emitc::mhlo::logical_or");
  patterns.add<CallOpConversion<mhlo::XorOp>>(ctx, "emitc::mhlo::logical_xor");

  // Insert patterns for MHLO tuple ops.
  patterns.add<CompareOpConversion>(ctx);
  patterns.add<CallOpConversion<mhlo::TupleOp>>(ctx, "std::make_tuple");
  patterns.add<GetTupleElementOpConversion>(ctx);

  // Insert patterns for MHLO slice ops.
  patterns.add<SliceOpConversion>(ctx);
  patterns.add<DynamicSliceOpConversion>(ctx);
  patterns.add<DynamicUpdateSliceOpConversion>(ctx);

  // Insert patterns for other MHLO ops.
  patterns.add<BatchNormInferenceOpConversion>(ctx);
  patterns.add<CallOpConversion<mhlo::BitcastConvertOp>>(
      ctx, "emitc::mhlo::bitcast_convert", /*explicitResultType=*/true);
  patterns.add<BroadcastInDimOpConversion>(ctx);
  patterns.add<CallOpConversion<mhlo::ClampOp>>(ctx, "emitc::mhlo::clamp",
                                                /*explicitResultType=*/false,
                                                /*explicitOperandTypes=*/true);
  patterns.add<ConcatenateOpConversion>(ctx);
  patterns.add<ConvOpConversion>(ctx);
  patterns.add<CallOpConversion<mhlo::DotOp>>(ctx, "emitc::mhlo::dot",
                                              /*explicitResultType=*/true);
  patterns.add<PadOpConversion>(ctx);
  patterns.add<CallOpConversion<mhlo::ReshapeOp>>(ctx, "emitc::mhlo::reshape",
                                                  /*explicitResultType=*/true);
  patterns.add<CallOpConversion<mhlo::SelectOp>>(ctx, "emitc::mhlo::select");

  // Insert patterns for MHLO RNG ops.
  patterns.add<RngOpConversion>(ctx);
}

namespace {

struct ConvertMhloToEmitCPass
    : public ConvertMHLOToEmitCBase<ConvertMhloToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnOperation() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();

    // clang-format off
    // MHLO nullary ops
    target.addIllegalOp<mhlo::ConstantOp>();

    // MHLO unary elementwise ops.
    target.addIllegalOp<mhlo::AbsOp,
                        mhlo::CeilOp,
                        mhlo::ConvertOp,
                        mhlo::CosineOp,
                        mhlo::ExpOp,
                        mhlo::Expm1Op,
                        mhlo::FloorOp,
                        mhlo::IsFiniteOp,
                        mhlo::LogOp,
                        mhlo::Log1pOp,
                        mhlo::NegOp,
                        mhlo::RoundOp,
                        mhlo::SineOp,
                        mhlo::SqrtOp,
                        mhlo::TanhOp>();

    // MHLO binary elementwise ops.
    target.addIllegalOp<mhlo::AddOp,
                        mhlo::Atan2Op,
                        mhlo::DivOp,
                        mhlo::MaxOp,
                        mhlo::MinOp,
                        mhlo::MulOp,
                        mhlo::PowOp,
                        mhlo::ShiftLeftOp,
                        mhlo::ShiftRightLogicalOp,
                        mhlo::SubtractOp>();

    // MHLO binary logical elementwise ops.
    target.addIllegalOp<mhlo::OrOp,
                        mhlo::XorOp>();

    // MHLO tuple ops.
    target.addIllegalOp<mhlo::CompareOp,
                        mhlo::TupleOp,
                        mhlo::GetTupleElementOp>();

    // MHLO slice ops.
    target.addIllegalOp<mhlo::DynamicSliceOp,
                        mhlo::DynamicUpdateSliceOp,
                        mhlo::SliceOp>();

    // MHLO region ops.
    target.addIllegalOp<mhlo::ReduceOp,
                        mhlo::ReturnOp>();

    // Other MHLO ops.
    target.addIllegalOp<mhlo::BatchNormInferenceOp,
                        mhlo::BitcastConvertOp,
                        mhlo::BroadcastInDimOp,
                        mhlo::ClampOp,
                        mhlo::ConcatenateOp,
                        mhlo::ConvolutionOp,
                        mhlo::DotOp,
                        mhlo::PadOp,
                        mhlo::ReshapeOp,
                        mhlo::SelectOp>();

    // MHLO RNG ops.
    target.addIllegalOp<mhlo::RngOp>();
    // clang-format on

    RewritePatternSet patterns(&getContext());
    populateMhloToEmitcPatterns(&getContext(), patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::emitc::createConvertMhloToEmitCPass() {
  return std::make_unique<ConvertMhloToEmitCPass>();
}
