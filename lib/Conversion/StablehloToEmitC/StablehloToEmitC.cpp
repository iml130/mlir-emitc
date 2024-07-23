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
#include "emitc/Conversion/EmitCCommon/GenericOpConversion.h"
#include "emitc/Conversion/StablehloToEmitC/StablehloToEmitC.h"

using namespace mlir;
using namespace mlir::emitc;

namespace {

/// Common functions.
SmallVector<Attribute, 2> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::to_vector<2>(
      llvm::map_range(llvm::seq<int64_t>(0, n), [&ctx](int64_t i) -> Attribute {
        return IntegerAttr::get(IndexType::get(ctx), i);
      }));
}

/// Convert `stablehlo.constant` into an `emitc.constant` operation.
class ConstOpConversion : public OpRewritePattern<stablehlo::ConstantOp> {
public:
  using OpRewritePattern<stablehlo::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConstantOp constOp,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, constOp.getType(),
                                                   constOp.getValue());
    return success();
  }
};

/// Convert `stablehlo.batch_norm_inference` into an `emitc.call_opaque`
/// operation.
class BatchNormInferenceOpConversion
    : public OpConversionPattern<stablehlo::BatchNormInferenceOp> {

public:
  BatchNormInferenceOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::BatchNormInferenceOp batchNormInferenceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "emitc::stablehlo::batch_norm_inference";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments = indexSequence(
        adaptor.getOperands().size(), batchNormInferenceOp.getContext());

    arguments.push_back(batchNormInferenceOp.getEpsilonAttr());
    arguments.push_back(batchNormInferenceOp.getFeatureIndexAttr());

    ArrayAttr args = rewriter.getArrayAttr(arguments);
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(batchNormInferenceOp.getResult().getType()),
         TypeAttr::get(adaptor.getScale().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        batchNormInferenceOp, batchNormInferenceOp.getType(), callee, args,
        templateArgs, adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.broadcast_in_dim` into an `emitc.call_opaque` operation.
class BroadcastInDimOpConversion
    : public OpConversionPattern<stablehlo::BroadcastInDimOp> {

public:
  BroadcastInDimOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::BroadcastInDimOp broadcastInDimOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::stablehlo::broadcast_in_dim";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments = indexSequence(
        adaptor.getOperands().size(), broadcastInDimOp.getContext());

    arguments.push_back(
        rewriter.getI64TensorAttr(broadcastInDimOp.getBroadcastDimensions()));

    ArrayAttr args = rewriter.getArrayAttr(arguments);

    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(broadcastInDimOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        broadcastInDimOp, broadcastInDimOp.getType(), callee, args,
        templateArgs, adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.concatenate` into an `emitc.call_opaque` operation.
class ConcatenateOpConversion
    : public OpConversionPattern<stablehlo::ConcatenateOp> {

public:
  ConcatenateOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::ConcatenateOp concatenateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "emitc::stablehlo::concatenate";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {rewriter.getI64IntegerAttr(concatenateOp.getDimension()),
         TypeAttr::get(concatenateOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        concatenateOp, concatenateOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.convolution` into an `emitc.call_opaque` operation.
class ConvOpConversion : public OpConversionPattern<stablehlo::ConvolutionOp> {

public:
  ConvOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::ConvolutionOp convOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "emitc::stablehlo::convolution";
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
    arguments.push_back(convOp.getPadding().value_or(
        rewriter.getI64TensorAttr(SmallVector<int64_t>(2, 0))));
    arguments.push_back(rewriter.getI64TensorAttr(
        convOp.getLhsDilation().value_or((SmallVector<int64_t>(2, 1)))));
    arguments.push_back(rewriter.getI64TensorAttr(
        convOp.getRhsDilation().value_or((SmallVector<int64_t>(2, 1)))));
    arguments.push_back(rewriter.getI64TensorAttr(
        convOp.getWindowStrides().value_or((SmallVector<int64_t>(2, 1)))));

    ArrayAttr args = rewriter.getArrayAttr(arguments);
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(convOp.getResult().getType()),
                               TypeAttr::get(adaptor.getLhs().getType()),
                               TypeAttr::get(adaptor.getRhs().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(convOp, convOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.compare` into an `emitc.call_opaque` operation.
class CompareOpConversion : public OpConversionPattern<stablehlo::CompareOp> {
  using OpConversionPattern<stablehlo::CompareOp>::OpConversionPattern;

public:
  CompareOpConversion(MLIRContext *ctx)
      : OpConversionPattern<stablehlo::CompareOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::CompareOp compareOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = compareOp.getContext();

    StringAttr callee = rewriter.getStringAttr("emitc::stablehlo::compare");

    stablehlo::ComparisonDirection comparisonDirection =
        compareOp.getComparisonDirection();
    std::optional<StringRef> functionName =
        StringSwitch<std::optional<StringRef>>(
            stringifyComparisonDirection(comparisonDirection))
            .Case("EQ", StringRef("std::equal_to"))
            .Case("NE", StringRef("std::not_equal_to"))
            .Case("GE", StringRef("std::greater_equal"))
            .Case("GT", StringRef("std::greater"))
            .Case("LE", StringRef("std::less_equal"))
            .Case("LT", StringRef("std::less"))
            .Default(std::nullopt);

    if (!functionName.has_value())
      return failure();

    Type elementType = compareOp.getOperand(0).getType();
    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(elementType),
         emitc::OpaqueAttr::get(ctx, functionName.value())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        compareOp, compareOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.get_tuple_element` into an `emitc.call_opaque` operation.
class GetTupleElementOpConversion
    : public OpConversionPattern<stablehlo::GetTupleElementOp> {
  using OpConversionPattern<stablehlo::GetTupleElementOp>::OpConversionPattern;

public:
  GetTupleElementOpConversion(MLIRContext *ctx)
      : OpConversionPattern<stablehlo::GetTupleElementOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::GetTupleElementOp getTupleElementOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto index = getTupleElementOp.getIndex();

    StringAttr callee = rewriter.getStringAttr("std::get");

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {IntegerAttr::get(rewriter.getIntegerType(32), index)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        getTupleElementOp, getTupleElementOp.getType(), callee, args,
        templateArgs, adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.slice` into an `emitc.call_opaque` operation.
class SliceOpConversion : public OpConversionPattern<stablehlo::SliceOp> {
  using OpConversionPattern<stablehlo::SliceOp>::OpConversionPattern;

public:
  SliceOpConversion(MLIRContext *ctx)
      : OpConversionPattern<stablehlo::SliceOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::SliceOp sliceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::stablehlo::slice";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments =
        indexSequence(adaptor.getOperands().size(), sliceOp.getContext());

    arguments.push_back(rewriter.getI64TensorAttr(sliceOp.getStartIndices()));
    arguments.push_back(rewriter.getI64TensorAttr(sliceOp.getLimitIndices()));
    arguments.push_back(rewriter.getI64TensorAttr(sliceOp.getStrides()));

    ArrayAttr args = rewriter.getArrayAttr(arguments);

    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(sliceOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(sliceOp, sliceOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.dynamic_slice` into an `emitc.call_opaque` operation.
class DynamicSliceOpConversion
    : public OpConversionPattern<stablehlo::DynamicSliceOp> {
  using OpConversionPattern<stablehlo::DynamicSliceOp>::OpConversionPattern;

public:
  DynamicSliceOpConversion(MLIRContext *ctx)
      : OpConversionPattern<stablehlo::DynamicSliceOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::DynamicSliceOp dynamicSliceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::stablehlo::dynamic_slice";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments = indexSequence(
        adaptor.getOperands().size(), dynamicSliceOp.getContext());

    arguments.push_back(
        rewriter.getI64TensorAttr(dynamicSliceOp.getSliceSizes()));

    ArrayAttr args = rewriter.getArrayAttr(arguments);

    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {TypeAttr::get(dynamicSliceOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        dynamicSliceOp, dynamicSliceOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.dynamic_update_slice` into an `emitc.call_opaque`
/// operation.
class DynamicUpdateSliceOpConversion
    : public OpConversionPattern<stablehlo::DynamicUpdateSliceOp> {
  using OpConversionPattern<
      stablehlo::DynamicUpdateSliceOp>::OpConversionPattern;

public:
  DynamicUpdateSliceOpConversion(MLIRContext *ctx)
      : OpConversionPattern<stablehlo::DynamicUpdateSliceOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::DynamicUpdateSliceOp dynamicUpdateSliceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "emitc::stablehlo::dynamic_update_slice";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(adaptor.getUpdate().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        dynamicUpdateSliceOp, dynamicUpdateSliceOp.getType(), callee, args,
        templateArgs, adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.pad` into an `emitc.call_opaque` operation.
class PadOpConversion : public OpConversionPattern<stablehlo::PadOp> {
  using OpConversionPattern<stablehlo::PadOp>::OpConversionPattern;

public:
  PadOpConversion(MLIRContext *ctx)
      : OpConversionPattern<stablehlo::PadOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::PadOp padOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::stablehlo::pad");

    SmallVector<Attribute, 2> arguments =
        indexSequence(adaptor.getOperands().size(), padOp.getContext());

    arguments.push_back(rewriter.getI64TensorAttr(padOp.getEdgePaddingLow()));
    arguments.push_back(rewriter.getI64TensorAttr(padOp.getEdgePaddingHigh()));
    arguments.push_back(rewriter.getI64TensorAttr(padOp.getInteriorPadding()));

    ArrayAttr args = rewriter.getArrayAttr(arguments);

    Type resultType = padOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(padOp, padOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.transpose` into an `emitc.call_opaque` operation.
class TransposeOpConversion
    : public OpConversionPattern<stablehlo::TransposeOp> {
  using OpConversionPattern<stablehlo::TransposeOp>::OpConversionPattern;

public:
  TransposeOpConversion(MLIRContext *ctx)
      : OpConversionPattern<stablehlo::TransposeOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::TransposeOp transposeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::stablehlo::transpose");

    SmallVector<Attribute> arguments =
        indexSequence(adaptor.getOperands().size(), transposeOp.getContext());

    arguments.push_back(
        rewriter.getI64TensorAttr(transposeOp.getPermutation()));
    ArrayAttr args = rewriter.getArrayAttr(arguments);

    Type resultType = transposeOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        transposeOp, transposeOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.rng` into an `emitc.call_opaque` operation.
class RngOpConversion : public OpConversionPattern<stablehlo::RngOp> {

public:
  RngOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::RngOp rngOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (rngOp.getRngDistribution() != stablehlo::RngDistribution::UNIFORM) {
      return rngOp.emitError(
          "Distributions other than uniform are not supported.");
    }

    StringRef funcName = "emitc::stablehlo::rng_uniform";
    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args;

    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(rngOp.getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(rngOp, rngOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());

    return success();
  }
};

/// Convert `stablehlo.rsqrt` into an `emitc.call_opaque` operation.
class RsqrtOpConversion : public OpConversionPattern<stablehlo::RsqrtOp> {
  using OpConversionPattern<stablehlo::RsqrtOp>::OpConversionPattern;

public:
  RsqrtOpConversion(MLIRContext *ctx)
      : OpConversionPattern<stablehlo::RsqrtOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(stablehlo::RsqrtOp rsqrtOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayAttr args;
    ArrayAttr templateArgs;

    // Create sqrt op.
    StringRef sqrtFuncName = "emitc::sqrt";
    StringAttr sqrtCallee = rewriter.getStringAttr(sqrtFuncName);

    auto sqrtEmitCOp = rewriter.create<emitc::CallOpaqueOp>(
        rsqrtOp.getLoc(), rsqrtOp.getType(), sqrtCallee, args, templateArgs,
        adaptor.getOperands());

    // Create reciprocal op.
    StringRef reciprocalFuncName = "emitc::stablehlo::rsqrt";
    StringAttr reciprocalCallee = rewriter.getStringAttr(reciprocalFuncName);

    auto reciprocalOp = rewriter.create<emitc::CallOpaqueOp>(
        sqrtEmitCOp.getLoc(), rsqrtOp.getType(), reciprocalCallee, args,
        templateArgs, sqrtEmitCOp.getResults());

    rewriter.replaceOp(rsqrtOp, reciprocalOp.getResults());

    return success();
  }
};

} // namespace

void populateStablehloToEmitcPatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns) {
  // Insert patterns for StableHLO nullary ops.
  patterns.add<ConstOpConversion>(ctx);

  // Insert patterns for StableHLO unary elementwise ops.
  patterns.add<GenericOpConversion<stablehlo::AbsOp>>(ctx,
                                                      "emitc::stablehlo::abs");
  patterns.add<GenericOpConversion<stablehlo::CeilOp>>(
      ctx, "emitc::stablehlo::ceil");
  patterns.add<GenericOpConversion<stablehlo::ConvertOp>>(
      ctx, "emitc::stablehlo::convert",
      /*explicitResultType=*/true);
  patterns.add<GenericOpConversion<stablehlo::CosineOp>>(
      ctx, "emitc::stablehlo::cos");
  patterns.add<GenericOpConversion<stablehlo::ExpOp>>(
      ctx, "emitc::stablehlo::exponential");
  patterns.add<GenericOpConversion<stablehlo::Expm1Op>>(
      ctx, "emitc::stablehlo::exponential_minus_one");
  patterns.add<GenericOpConversion<stablehlo::FloorOp>>(
      ctx, "emitc::stablehlo::floor");
  patterns.add<GenericOpConversion<stablehlo::IsFiniteOp>>(
      ctx, "emitc::stablehlo::is_finite");
  patterns.add<GenericOpConversion<stablehlo::LogOp>>(ctx,
                                                      "emitc::stablehlo::log");
  patterns.add<GenericOpConversion<stablehlo::Log1pOp>>(
      ctx, "emitc::stablehlo::log_plus_one");
  patterns.add<GenericOpConversion<stablehlo::NegOp>>(
      ctx, "emitc::stablehlo::negate");
  patterns.add<GenericOpConversion<stablehlo::RoundOp>>(
      ctx, "emitc::stablehlo::round");
  patterns.add<GenericOpConversion<stablehlo::SineOp>>(ctx,
                                                       "emitc::stablehlo::sin");
  patterns.add<GenericOpConversion<stablehlo::SqrtOp>>(
      ctx, "emitc::stablehlo::sqrt");
  patterns.add<GenericOpConversion<stablehlo::RsqrtOp>>(
      ctx, "emitc::stablehlo::rsqrt");
  patterns.add<GenericOpConversion<stablehlo::TanhOp>>(
      ctx, "emitc::stablehlo::tanh");

  // Insert patterns for StableHLO binary elementwise ops.
  patterns.add<GenericOpConversion<stablehlo::AddOp>>(ctx,
                                                      "emitc::stablehlo::add");
  patterns.add<GenericOpConversion<stablehlo::Atan2Op>>(
      ctx, "emitc::stablehlo::atan2");
  patterns.add<GenericOpConversion<stablehlo::DivOp>>(ctx,
                                                      "emitc::stablehlo::div");
  patterns.add<GenericOpConversion<stablehlo::MaxOp>>(ctx,
                                                      "emitc::stablehlo::max");
  patterns.add<GenericOpConversion<stablehlo::MinOp>>(ctx,
                                                      "emitc::stablehlo::min");
  patterns.add<GenericOpConversion<stablehlo::MulOp>>(ctx,
                                                      "emitc::stablehlo::mul");
  patterns.add<GenericOpConversion<stablehlo::PowOp>>(ctx,
                                                      "emitc::stablehlo::pow");
  patterns.add<GenericOpConversion<stablehlo::ShiftLeftOp>>(
      ctx, "emitc::stablehlo::shift_left");
  patterns.add<GenericOpConversion<stablehlo::ShiftRightLogicalOp>>(
      ctx, "emitc::stablehlo::shift_right_logical");
  patterns.add<GenericOpConversion<stablehlo::SubtractOp>>(
      ctx, "emitc::stablehlo::sub");

  // Insert patterns for StableHLO binary logical elementwise ops.
  patterns.add<GenericOpConversion<stablehlo::OrOp>>(
      ctx, "emitc::stablehlo::logical_or");
  patterns.add<GenericOpConversion<stablehlo::XorOp>>(
      ctx, "emitc::stablehlo::logical_xor");

  // Insert patterns for StableHLO tuple ops.
  patterns.add<CompareOpConversion>(ctx);
  patterns.add<GenericOpConversion<stablehlo::TupleOp>>(ctx, "std::make_tuple");
  patterns.add<GetTupleElementOpConversion>(ctx);

  // Insert patterns for StableHLO slice ops.
  patterns.add<SliceOpConversion>(ctx);
  patterns.add<DynamicSliceOpConversion>(ctx);
  patterns.add<DynamicUpdateSliceOpConversion>(ctx);

  // Insert patterns for other StableHLO ops.
  patterns.add<BatchNormInferenceOpConversion>(ctx);
  patterns.add<GenericOpConversion<stablehlo::BitcastConvertOp>>(
      ctx, "emitc::stablehlo::bitcast_convert", /*explicitResultType=*/true);
  patterns.add<BroadcastInDimOpConversion>(ctx);
  patterns.add<GenericOpConversion<stablehlo::ClampOp>>(
      ctx, "emitc::stablehlo::clamp",
      /*explicitResultType=*/false,
      /*explicitOperandTypes=*/true);
  patterns.add<ConcatenateOpConversion>(ctx);
  patterns.add<ConvOpConversion>(ctx);
  patterns.add<GenericOpConversion<stablehlo::DotOp>>(
      ctx, "emitc::stablehlo::dot",
      /*explicitResultType=*/true);
  patterns.add<PadOpConversion>(ctx);
  patterns.add<GenericOpConversion<stablehlo::ReshapeOp>>(
      ctx, "emitc::stablehlo::reshape",
      /*explicitResultType=*/true);
  patterns.add<GenericOpConversion<stablehlo::SelectOp>>(
      ctx, "emitc::stablehlo::select");
  patterns.add<TransposeOpConversion>(ctx);

  // Insert patterns for StableHLO RNG ops.
  patterns.add<RngOpConversion>(ctx);
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
    // StableHLO nullary ops
    target.addIllegalOp<stablehlo::ConstantOp>();

    // StableHLO unary elementwise ops.
    target.addIllegalOp<stablehlo::AbsOp,
                        stablehlo::CeilOp,
                        stablehlo::ConvertOp,
                        stablehlo::CosineOp,
                        stablehlo::ExpOp,
                        stablehlo::Expm1Op,
                        stablehlo::FloorOp,
                        stablehlo::IsFiniteOp,
                        stablehlo::LogOp,
                        stablehlo::Log1pOp,
                        stablehlo::NegOp,
                        stablehlo::RoundOp,
                        stablehlo::SineOp,
                        stablehlo::SqrtOp,
                        stablehlo::RsqrtOp,
                        stablehlo::TanhOp>();

    // StableHLO binary elementwise ops.
    target.addIllegalOp<stablehlo::AddOp,
                        stablehlo::Atan2Op,
                        stablehlo::DivOp,
                        stablehlo::MaxOp,
                        stablehlo::MinOp,
                        stablehlo::MulOp,
                        stablehlo::PowOp,
                        stablehlo::ShiftLeftOp,
                        stablehlo::ShiftRightLogicalOp,
                        stablehlo::SubtractOp>();

    // StableHLO binary logical elementwise ops.
    target.addIllegalOp<stablehlo::OrOp,
                        stablehlo::XorOp>();

    // StableHLO tuple ops.
    target.addIllegalOp<stablehlo::CompareOp,
                        stablehlo::TupleOp,
                        stablehlo::GetTupleElementOp>();

    // StableHLO slice ops.
    target.addIllegalOp<stablehlo::DynamicSliceOp,
                        stablehlo::DynamicUpdateSliceOp,
                        stablehlo::SliceOp>();

    // StableHLO region ops.
    target.addIllegalOp<stablehlo::ReduceOp,
                        stablehlo::ReturnOp>();

    // Other StableHLO ops.
    target.addIllegalOp<stablehlo::BatchNormInferenceOp,
                        stablehlo::BitcastConvertOp,
                        stablehlo::BroadcastInDimOp,
                        stablehlo::ClampOp,
                        stablehlo::ConcatenateOp,
                        stablehlo::ConvolutionOp,
                        stablehlo::DotOp,
                        stablehlo::PadOp,
                        stablehlo::ReshapeOp,
                        stablehlo::SelectOp,
                        stablehlo::TransposeOp>();

    // StableHLO RNG ops.
    target.addIllegalOp<stablehlo::RngOp>();
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
