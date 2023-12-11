//===- TosaToEmitC.cpp - TOSA to EmitC conversion -------------------------===//
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

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"
#include "emitc/Conversion/EmitCCommon/GenericOpConversion.h"
#include "emitc/Conversion/TosaToEmitC/TosaToEmitC.h"

using namespace mlir;
using namespace mlir::emitc;

namespace {

/// Common functions.
DenseIntElementsAttr getI64ElementsAttr(const ArrayRef<long> values,
                                        MLIRContext *ctx) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, IntegerType::get(ctx, 64));

  return DenseIntElementsAttr::get(ty, values);
}

SmallVector<Attribute, 2> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::to_vector<2>(
      llvm::map_range(llvm::seq<int64_t>(0, n), [&ctx](int64_t i) -> Attribute {
        return IntegerAttr::get(IndexType::get(ctx), i);
      }));
}

/// Convert `tosa.concat` into an `emitc.call_opaque` operation.
class ConcatOpConversion : public OpConversionPattern<tosa::ConcatOp> {

public:
  ConcatOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::ConcatOp concatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "emitc::tosa::concat";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({concatOp.getAxisAttr(),
                               TypeAttr::get(concatOp.getResult().getType())});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        concatOp, concatOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};

/// Convert `tosa.const` into an `emitc.constant` operation.
class ConstOpConversion : public OpRewritePattern<tosa::ConstOp> {
public:
  using OpRewritePattern<tosa::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConstOp constOp,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, constOp.getType(),
                                                   constOp.getValue());
    return success();
  }
};

/// Convert a common `tosa` convolution operation into an `emitc.call_opaque`
/// operation.
template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class GenericConvOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  GenericConvOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp convOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Fail if quantization is requested.
    if (convOp.getQuantizationInfo().has_value()) {
      return convOp.emitError("Quantization for " + convOp.getOperationName() +
                              " is currently not supported.");
    }

    // All tosa conv* ops support adding a bias after the actual convolution. We
    // will split the convolution with bias into two operations: the convolution
    // (as an emitc call op) and the addition of the bias (as an tosa.add op).
    // Therefore we remove bias from the operands of the convolution.
    auto operands = adaptor.getOperands();
    operands = operands.drop_back();

    StringAttr callee = rewriter.getStringAttr(funcName);

    // clang-format off
    ArrayAttr args = rewriter.getArrayAttr({
      rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(1),
      getI64ElementsAttr(convOp.getPad(), convOp.getContext()),
      getI64ElementsAttr(convOp.getStride(), convOp.getContext()),
      getI64ElementsAttr(convOp.getDilation(), convOp.getContext()),
    });
    // clang-format on

    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(convOp.getResult().getType())});

    // Create conv op.
    auto emitcConvOp = rewriter.create<emitc::CallOpaqueOp>(
        convOp->getLoc(), convOp.getType(), callee, args, templateArgs,
        operands);

    auto output = emitcConvOp.getResult(0);
    auto tosaAddOp = rewriter.create<tosa::AddOp>(
        convOp.getLoc(), output.getType(), output, convOp.getBias());

    rewriter.replaceOp(convOp, {tosaAddOp.getResult()});

    return success();
  }

  StringRef funcName;
};

/// Convert a common `tosa` pooling operation into an `emitc.call_opaque`
/// operation.
template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class GenericPoolOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  GenericPoolOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp poolOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr(funcName);

    // TODO: average pool has an acc_type attribute.
    // clang-format off
    ArrayAttr args = rewriter.getArrayAttr({
      rewriter.getIndexAttr(0),
      getI64ElementsAttr(poolOp.getPad(), poolOp.getContext()),
      getI64ElementsAttr(poolOp.getStride(), poolOp.getContext()),
      getI64ElementsAttr(poolOp.getKernel(), poolOp.getContext()),
    });
    // clang-format on

    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(poolOp.getResult().getType())});

    // Create pool op.
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(poolOp, poolOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());

    return success();
  }

  StringRef funcName;
};

/// Convert `tosa.fully_connected` into an `emitc.call_opaque` operation.
class FullyConnectedOpConversion
    : public OpConversionPattern<tosa::FullyConnectedOp> {
  using OpConversionPattern<tosa::FullyConnectedOp>::OpConversionPattern;

public:
  FullyConnectedOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<tosa::FullyConnectedOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::FullyConnectedOp fullyConnectedOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (fullyConnectedOp.getQuantizationInfo().has_value()) {
      return fullyConnectedOp.emitError(
          "Quantization of tosa.fully_connected is currently not supported.");
    }

    StringRef funcName = "emitc::tosa::fully_connected";
    StringAttr callee = rewriter.getStringAttr(funcName);

    Type type = fullyConnectedOp.getType();

    ArrayAttr args;
    ArrayAttr templateArgs =
        ArrayAttr::get(fullyConnectedOp.getContext(), {TypeAttr::get(type)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(fullyConnectedOp, type,
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());
    return success();
  }
};

/// Convert `tosa.matmul` into an `emitc.call_opaque` operation.
class MatMulOpConversion : public OpConversionPattern<tosa::MatMulOp> {
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;

public:
  MatMulOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::MatMulOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::MatMulOp matMulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (matMulOp.getQuantizationInfo().has_value()) {
      return matMulOp.emitError(
          "Quantization of tosa.matmul is currently not supported.");
    }

    StringRef funcName = "emitc::tosa::matmul";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        matMulOp, matMulOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());
    return success();
  }
};

/// Convert `tosa.clamp` into an `emitc.call_opaque` operation.
class ClampOpConversion : public OpConversionPattern<tosa::ClampOp> {
  using OpConversionPattern<tosa::ClampOp>::OpConversionPattern;

public:
  ClampOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::ClampOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::ClampOp clampOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "emitc::tosa::clamp";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> arguments;
    arguments.push_back(rewriter.getIndexAttr(0));

    // TOSA specifies the min/max attributes to be either exact i64 or f32,
    // regardless of the operand's element type. So we need to make sure that
    // the min/max attribute type match the operand's element type and it's bit
    // width.
    auto elementType =
        adaptor.getInput().getType().cast<RankedTensorType>().getElementType();
    if (elementType.isa<IntegerType>()) {
      // Change the {min,max}_int type to the element type of the operand.
      auto minInt = clampOp.getMinInt();
      auto maxInt = clampOp.getMaxInt();
      arguments.push_back(IntegerAttr::get(elementType, minInt));
      arguments.push_back(IntegerAttr::get(elementType, maxInt));
    } else if (elementType.isa<FloatType>()) {
      // Change the {min,max}_fp type to the element type of the operand.
      auto minFp = clampOp.getMinFpAttr().getValueAsDouble();
      auto maxFp = clampOp.getMaxFpAttr().getValueAsDouble();
      arguments.push_back(FloatAttr::get(elementType, minFp));
      arguments.push_back(FloatAttr::get(elementType, maxFp));
    } else {
      return clampOp.emitError(
          "Operand of tosa.clamp has to be tensor of integer or float values.");
    }
    ArrayAttr args = rewriter.getArrayAttr(arguments);
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(clampOp, clampOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());

    return success();
  }
};

/// Convert `tosa.negate` into an `emitc.call_opaque` operation.
class NegateOpConversion : public OpConversionPattern<tosa::NegateOp> {
  using OpConversionPattern<tosa::NegateOp>::OpConversionPattern;

public:
  NegateOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::NegateOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::NegateOp negateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (negateOp.getQuantizationInfo().has_value()) {
      return negateOp.emitError(
          "Quantization of tosa.negate is currently not supported.");
    }

    StringRef funcName = "emitc::tosa::negate";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        negateOp, negateOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());
    return success();
  }
};

/// Convert `tosa.rescale` into an `emitc.call_opaque` operation.
class RescaleOpConversion : public OpConversionPattern<tosa::RescaleOp> {
  using OpConversionPattern<tosa::RescaleOp>::OpConversionPattern;

public:
  RescaleOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::RescaleOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::RescaleOp rescaleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::tosa::rescale";
    StringAttr callee = rewriter.getStringAttr(funcName);

    // clang-format off
    ArrayAttr args = rewriter.getArrayAttr({
        rewriter.getIndexAttr(0),
        rescaleOp.getInputZpAttr(),
        rescaleOp.getOutputZpAttr(),
        rescaleOp.getMultiplierAttr(),
        rescaleOp.getShiftAttr(),
        rescaleOp.getScale32Attr(),
        rescaleOp.getDoubleRoundAttr(),
        rescaleOp.getPerChannelAttr()
    });
    // clang-format on

    TypeAttr resultType = TypeAttr::get(rescaleOp.getResult().getType());
    IntegerAttr arraySize =
        rewriter.getI32IntegerAttr(rescaleOp.getMultiplierAttr().size());
    ArrayAttr templateArgs = rewriter.getArrayAttr({resultType, arraySize});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        rescaleOp, rescaleOp.getType(), callee, args, templateArgs,
        adaptor.getOperands());

    return success();
  }
};

/// Convert `tosa.rsqrt` into an `emitc.call_opaque` operation.
class RsqrtOpConversion : public OpConversionPattern<tosa::RsqrtOp> {
  using OpConversionPattern<tosa::RsqrtOp>::OpConversionPattern;

public:
  RsqrtOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::RsqrtOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::RsqrtOp rsqrtOp, OpAdaptor adaptor,
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
    StringRef reciprocalFuncName = "emitc::tosa::reciprocal";
    StringAttr reciprocalCallee = rewriter.getStringAttr(reciprocalFuncName);

    auto reciprocalOp = rewriter.create<emitc::CallOpaqueOp>(
        sqrtEmitCOp.getLoc(), rsqrtOp.getType(), reciprocalCallee, args,
        templateArgs, sqrtEmitCOp.getResults());

    rewriter.replaceOp(rsqrtOp, reciprocalOp.getResults());

    return success();
  }
};

/// Creates a broadcast operation if the inputs don't match the output shape.
template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
SmallVector<Value>
createBroadcastOpIfNeeded(SrcOp &srcOp, Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) {
  // TOSA allows implicit broadcasting, so we need to insert broadcast_in_dim
  // ops if necessary, e.g.:
  // tensor<8xf32>     -> tensor<1x4x4x8xf32> Broadcast dims = (3)
  // tensor<4x8xf32>   -> tensor<1x4x4x8xf32> Broadcast dims = (2, 3)
  // tensor<4x4x8xf32> -> tensor<1x4x4x8xf32> Broadcast dims = (1, 2, 3)

  StringRef broadcastFuncName = "emitc::broadcast_in_dim";
  StringAttr broadcastCallee = rewriter.getStringAttr(broadcastFuncName);

  Value output = srcOp.getResult();
  auto opOutputShape = output.getType().cast<RankedTensorType>().getShape();
  auto opOutputRank = output.getType().cast<RankedTensorType>().getRank();
  SmallVector<Value> broadcastedOperands;

  for (auto operand : adaptor.getOperands()) {
    RankedTensorType operandTensor =
        operand.getType().template cast<RankedTensorType>();
    auto operandShape = operandTensor.getShape();
    auto operandRank = operandTensor.getRank();

    // Insert a broadcast_in_dim operation if shape of operands don't match.
    if (!operandShape.equals(opOutputShape)) {
      SmallVector<Attribute, 1> broadcastIndices;
      auto numBroadcastDims = opOutputRank - operandRank;
      for (int64_t d = numBroadcastDims; d < opOutputRank; ++d) {
        broadcastIndices.push_back(
            IntegerAttr::get(rewriter.getIntegerType(64), d));
      }

      RankedTensorType tensorType =
          RankedTensorType::get({static_cast<int64_t>(operandRank)},
                                IntegerType::get(srcOp.getContext(), 64));
      ArrayAttr broadcastArgs = rewriter.getArrayAttr(
          {rewriter.getIndexAttr(0),
           DenseIntElementsAttr::get(tensorType, broadcastIndices)});

      auto newBroadcastType = RankedTensorType::get(
          llvm::ArrayRef(opOutputShape), operandTensor.getElementType());

      ArrayAttr templateBroadcastArgs =
          rewriter.getArrayAttr({TypeAttr::get(newBroadcastType)});

      auto broadcastArg = rewriter.create<emitc::CallOpaqueOp>(
          srcOp->getLoc(), newBroadcastType, broadcastCallee, broadcastArgs,
          templateBroadcastArgs, operand);
      // Replace the original operand with the result of the broadcast_in_dim
      // operation.
      broadcastedOperands.push_back(broadcastArg.getResult(0));
    } else {
      // No broadcasting needed. Store original operand.
      broadcastedOperands.push_back(operand);
    }
  }
  return broadcastedOperands;
}

/// Convert a common, broadcastable `tosa` operation into an `emitc.call_opaque`
/// operation.
template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class CallOpBroadcastableConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  CallOpBroadcastableConversion(MLIRContext *ctx, StringRef funcName,
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

    SmallVector<Value, 2> broadcastedOperands =
        createBroadcastOpIfNeeded(srcOp, adaptor, rewriter);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, srcOp.getType(), callee, args, templateArgs,
        ValueRange({broadcastedOperands[0], broadcastedOperands[1]}));

    return success();
  }

  StringRef funcName;
  // If set, use the result type of the operation as template parameter.
  bool explicitResultType;
  // If set, use the operand types as (additional) template parameters.
  bool explicitOperandTypes;
};

/// Convert `tosa.mul` into an `emitc.call_opaque` operation.
class MulOpConversion : public OpConversionPattern<tosa::MulOp> {
  using OpConversionPattern<tosa::MulOp>::OpConversionPattern;

public:
  MulOpConversion(MLIRContext *ctx, StringRef funcName,
                  bool explicitResultType = false,
                  bool explicitOperandTypes = false)
      : OpConversionPattern<tosa::MulOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::MulOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::tosa::mul";
    StringAttr callee = rewriter.getStringAttr(funcName);

    auto shiftAttr = mulOp.getShiftAttr();
    ArrayAttr args;
    SmallVector<Attribute, 1> arguments;
    if (shiftAttr.getInt() > 0) {
      arguments.push_back(rewriter.getIndexAttr(0));
      arguments.push_back(rewriter.getIndexAttr(1));
      arguments.push_back(shiftAttr);
      args = rewriter.getArrayAttr(arguments);
    }

    ArrayAttr templateArgs;

    SmallVector<Value, 2> broadcastedOperands =
        createBroadcastOpIfNeeded(mulOp, adaptor, rewriter);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        mulOp, mulOp.getType(), callee, args, templateArgs,
        ValueRange({broadcastedOperands[0], broadcastedOperands[1]}));

    return success();
  }
};

/// Convert `tosa.arithmetic_right_shift` into an `emitc.call_opaque` operation.
class ArithmeticRightShiftOpConversion
    : public OpConversionPattern<tosa::ArithmeticRightShiftOp> {
  using OpConversionPattern<tosa::ArithmeticRightShiftOp>::OpConversionPattern;

public:
  ArithmeticRightShiftOpConversion(MLIRContext *ctx, StringRef funcName,
                                   bool explicitResultType = false,
                                   bool explicitOperandTypes = false)
      : OpConversionPattern<tosa::ArithmeticRightShiftOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::ArithmeticRightShiftOp arithmeticRightShiftOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::tosa::arithmetic_right_shift";
    StringAttr callee = rewriter.getStringAttr(funcName);

    auto roundAttr = arithmeticRightShiftOp.getRoundAttr();
    ArrayAttr args;
    SmallVector<Attribute, 1> arguments;
    arguments.push_back(rewriter.getIndexAttr(0));
    arguments.push_back(rewriter.getIndexAttr(1));
    arguments.push_back(roundAttr);
    args = rewriter.getArrayAttr(arguments);

    ArrayAttr templateArgs;

    SmallVector<Value, 2> broadcastedOperands =
        createBroadcastOpIfNeeded(arithmeticRightShiftOp, adaptor, rewriter);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        arithmeticRightShiftOp, arithmeticRightShiftOp.getType(), callee, args,
        templateArgs,
        ValueRange({broadcastedOperands[0], broadcastedOperands[1]}));

    return success();
  }
};

/// Convert `tosa.select` into an `emitc.call_opaque` operation.
class SelectOpConversion : public OpConversionPattern<tosa::SelectOp> {
  using OpConversionPattern<tosa::SelectOp>::OpConversionPattern;

public:
  SelectOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::SelectOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::SelectOp selectOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "emitc::tosa::select";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs;

    SmallVector<Value, 3> broadcastedOperands =
        createBroadcastOpIfNeeded(selectOp, adaptor, rewriter);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        selectOp, selectOp.getType(), callee, args, templateArgs,
        broadcastedOperands);

    return success();
  }
};

/// Convert `tosa.reduce_*` into an `emitc.call_opaque` operation.
template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class ReduceOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  ReduceOpConversion(MLIRContext *ctx, StringRef funcName, bool keepDims)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName),
        keepDims(keepDims) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp reduceOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute> arguments =
        indexSequence(adaptor.getOperands().size(), reduceOp.getContext());
    arguments.push_back(reduceOp.getAxisAttr());

    ArrayAttr args = rewriter.getArrayAttr(arguments);

    if (keepDims) {
      // We need to adjust output shape of reduce since our implementation does
      // not keep reduced dimensions.
      Value output = reduceOp.getResult();
      RankedTensorType reducedOutputType =
          output.getType().cast<RankedTensorType>();

      SmallVector<int64_t> newReducedOutputShape;

      for (auto dim : reducedOutputType.getShape()) {
        newReducedOutputShape.push_back(dim);
      };

      // Remove reduced axis from shape.
      newReducedOutputShape.erase(newReducedOutputShape.begin() +
                                  reduceOp.getAxis());

      auto newOutputType =
          RankedTensorType::get(llvm::ArrayRef(newReducedOutputShape),
                                reducedOutputType.getElementType());

      ArrayAttr templateArgs =
          rewriter.getArrayAttr({TypeAttr::get(newOutputType),
                                 TypeAttr::get(reduceOp.getInput().getType())});

      auto emitcReduceOp = rewriter.create<emitc::CallOpaqueOp>(
          reduceOp.getLoc(), newOutputType, callee, args, templateArgs,
          adaptor.getOperands());

      // Create tosa.reshape op.
      SmallVector<int64_t> newShapeAttr_;
      for (auto dim : output.getType().cast<RankedTensorType>().getShape()) {
        newShapeAttr_.push_back(dim);
      };

      DenseI64ArrayAttr newShapeAttr =
          DenseI64ArrayAttr::get(reduceOp.getContext(), newShapeAttr_);

      rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
          reduceOp, output.getType(), emitcReduceOp.getResult(0), newShapeAttr);
    } else {
      ArrayAttr templateArgs =
          rewriter.getArrayAttr({TypeAttr::get(reduceOp.getType()),
                                 TypeAttr::get(reduceOp.getInput().getType())});
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          reduceOp, reduceOp.getType(), callee, args, templateArgs,
          adaptor.getOperands());
    }

    return success();
  }

  StringRef funcName;
  bool keepDims;
};

/// Convert `tosa.pad` into an `emitc.call_opaque` operation.
class PadOpConversion : public OpConversionPattern<tosa::PadOp> {
  using OpConversionPattern<tosa::PadOp>::OpConversionPattern;

public:
  PadOpConversion(MLIRContext *ctx) : OpConversionPattern<tosa::PadOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::PadOp padOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (padOp.getQuantizationInfo().has_value()) {
      return padOp.emitError(
          "Quantization of tosa.pad is currently not supported.");
    }

    StringAttr callee = rewriter.getStringAttr("emitc::tosa::pad");

    // No arguments! Pad itself is an operand and not an argument. Therefore, we
    // have to handle any conversion in tosa::pad.
    ArrayAttr args;

    Type resultType = padOp.getOutput().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(padOp, padOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());

    return success();
  }
};

/// Convert `tosa.slice` into an `emitc.call_opaque` operation.
class SliceOpConversion : public OpConversionPattern<tosa::SliceOp> {
  using OpConversionPattern<tosa::SliceOp>::OpConversionPattern;

public:
  SliceOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::SliceOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::SliceOp sliceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::tosa::slice");

    // clang-format off
    ArrayAttr args = rewriter.getArrayAttr({
      rewriter.getIndexAttr(0),
      sliceOp.getStartAttr(),
      sliceOp.getSizeAttr(),
    });
    // clang-format on

    Type resultType = sliceOp.getOutput().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(sliceOp, sliceOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());

    return success();
  }
};

/// Convert `tosa.tile` into an `emitc.call_opaque` operation.
class TileOpConversion : public OpConversionPattern<tosa::TileOp> {
  using OpConversionPattern<tosa::TileOp>::OpConversionPattern;

public:
  TileOpConversion(MLIRContext *ctx) : OpConversionPattern<tosa::TileOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::TileOp tileOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::tosa::tile");
    auto inputShape =
        adaptor.getInput1().getType().cast<RankedTensorType>().getShape();
    for (int64_t i = 0, e = inputShape.size(); i < e; i++) {
      if (inputShape[i] > std::numeric_limits<int>::max()) {
        return tileOp.emitError("tosa.tile with dimensions larger than the "
                                "i32 limit are not supported.");
      }
    }

    // clang-format off
    ArrayAttr args = rewriter.getArrayAttr({
      rewriter.getIndexAttr(0),
      tileOp.getMultiplesAttr(),
    });
    // clang-format on

    Type resultType = tileOp.getOutput().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(tileOp, tileOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());
    return success();
  }
};

} // namespace

void populateTosaToEmitcPatterns(MLIRContext *ctx,
                                 RewritePatternSet &patterns) {
  // Insert patterns for TOSA data node ops.
  patterns.add<ConstOpConversion>(ctx);

  // Insert patterns for TOSA unary elementwise ops.
  patterns.add<GenericOpConversion<tosa::AbsOp>>(ctx, "emitc::tosa::abs");
  patterns.add<GenericOpConversion<tosa::CastOp>>(ctx, "emitc::tosa::cast",
                                                  /*explicitResultType=*/true);
  patterns.add<GenericOpConversion<tosa::CeilOp>>(ctx, "emitc::tosa::ceil");
  patterns.add<GenericOpConversion<tosa::ClzOp>>(ctx, "emitc::tosa::clz");
  patterns.add<ClampOpConversion>(ctx);
  patterns.add<GenericOpConversion<tosa::ExpOp>>(ctx, "emitc::tosa::exp");
  patterns.add<GenericOpConversion<tosa::FloorOp>>(ctx, "emitc::tosa::floor");
  patterns.add<GenericOpConversion<tosa::LogOp>>(ctx, "emitc::tosa::log");
  patterns.add<NegateOpConversion>(ctx);
  patterns.add<GenericOpConversion<tosa::ReciprocalOp>>(
      ctx, "emitc::tosa::reciprocal");
  patterns.add<RescaleOpConversion>(ctx);
  patterns.add<RsqrtOpConversion>(ctx);
  patterns.add<GenericOpConversion<tosa::TanhOp>>(ctx, "emitc::tosa::tanh");

  // Insert patterns for TOSA binary elementwise ops.
  patterns.add<CallOpBroadcastableConversion<tosa::AddOp>>(ctx,
                                                           "emitc::tosa::add");
  patterns.add<ArithmeticRightShiftOpConversion>(
      ctx, "emitc::tosa::arithmetic_right_shift");
  patterns.add<CallOpBroadcastableConversion<tosa::EqualOp>>(
      ctx, "emitc::tosa::equal", /*explicitResultType=*/true);
  patterns.add<CallOpBroadcastableConversion<tosa::GreaterEqualOp>>(
      ctx, "emitc::tosa::greater_equal", /*explicitResultType=*/true);
  patterns.add<CallOpBroadcastableConversion<tosa::LogicalLeftShiftOp>>(
      ctx, "emitc::tosa::logical_left_shift");
  patterns.add<CallOpBroadcastableConversion<tosa::MaximumOp>>(
      ctx, "emitc::tosa::maximum");
  patterns.add<CallOpBroadcastableConversion<tosa::MinimumOp>>(
      ctx, "emitc::tosa::minimum");
  patterns.add<MulOpConversion>(ctx, "emitc::tosa::mul");
  patterns.add<CallOpBroadcastableConversion<tosa::PowOp>>(ctx,
                                                           "emitc::tosa::pow");
  patterns.add<CallOpBroadcastableConversion<tosa::SubOp>>(ctx,
                                                           "emitc::tosa::sub");
  patterns.add<GenericOpConversion<tosa::TableOp>>(ctx, "emitc::tosa::table");

  // Insert patterns for TOSA ternary elementwise ops.
  patterns.add<SelectOpConversion>(ctx);

  // Insert patterns for other TOSA ops.
  patterns.add<ConcatOpConversion>(ctx);
  patterns.add<GenericConvOpConversion<tosa::Conv2DOp>>(ctx,
                                                        "emitc::tosa::conv2d");
  patterns.add<GenericConvOpConversion<tosa::DepthwiseConv2DOp>>(
      ctx, "emitc::tosa::depthwise_conv2d");
  patterns.add<GenericPoolOpConversion<tosa::AvgPool2dOp>>(
      ctx, "emitc::tosa::avg_pool2d");
  patterns.add<GenericPoolOpConversion<tosa::MaxPool2dOp>>(
      ctx, "emitc::tosa::max_pool2d");
  patterns.add<FullyConnectedOpConversion>(ctx, "emitc::tosa::fully_connected");
  patterns.add<GenericOpConversion<tosa::GatherOp>>(
      ctx, "emitc::tosa::gather",
      /*explicitResultType=*/true);
  patterns.add<MatMulOpConversion>(ctx);
  patterns.add<TileOpConversion>(ctx);
  patterns.add<ReduceOpConversion<tosa::ArgMaxOp>>(ctx, "emitc::tosa::argmax",
                                                   false);
  patterns.add<ReduceOpConversion<tosa::ReduceAllOp>>(
      ctx, "emitc::tosa::reduce_all", true);
  patterns.add<ReduceOpConversion<tosa::ReduceAnyOp>>(
      ctx, "emitc::tosa::reduce_any", true);
  patterns.add<ReduceOpConversion<tosa::ReduceMaxOp>>(
      ctx, "emitc::tosa::reduce_max", true);
  patterns.add<ReduceOpConversion<tosa::ReduceMinOp>>(
      ctx, "emitc::tosa::reduce_min", true);
  patterns.add<ReduceOpConversion<tosa::ReduceProdOp>>(
      ctx, "emitc::tosa::reduce_prod", true);
  patterns.add<ReduceOpConversion<tosa::ReduceSumOp>>(
      ctx, "emitc::tosa::reduce_sum", true);
  patterns.add<GenericOpConversion<tosa::ReshapeOp>>(
      ctx, "emitc::tosa::reshape",
      /*explicitResultType=*/true);
  patterns.add<SliceOpConversion>(ctx);
  patterns.add<PadOpConversion>(ctx);
  patterns.add<GenericOpConversion<tosa::TransposeOp>>(
      ctx, "emitc::tosa::transpose", /*explicitResultType=*/true);
}

namespace {

struct ConvertTosaToEmitCPass
    : public ConvertTosaToEmitCBase<ConvertTosaToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnOperation() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<tosa::TosaDialect>();

    // clang-format off
    // Data node ops.
    target.addIllegalOp<tosa::ConstOp>();

    // Unary elementwise ops.
    target.addIllegalOp<tosa::AbsOp,
                        tosa::CastOp,
                        tosa::CeilOp,
                        tosa::ClampOp,
                        tosa::ClzOp,
                        tosa::ExpOp,
                        tosa::FloorOp,
                        tosa::LogOp,
                        tosa::NegateOp,
                        tosa::ReciprocalOp,
                        tosa::RescaleOp,
                        tosa::RsqrtOp,
                        tosa::TanhOp>();

    // Binary elementwise ops.
    target.addIllegalOp<tosa::AddOp,
                        tosa::ArithmeticRightShiftOp,
                        tosa::EqualOp,
                        tosa::GreaterEqualOp,
                        tosa::LogicalLeftShiftOp,
                        tosa::MaximumOp,
                        tosa::MinimumOp,
                        tosa::MulOp,
                        tosa::PowOp,
                        tosa::SubOp,
                        tosa::TableOp>();

    // Ternary elementwise ops.
    target.addIllegalOp<tosa::SelectOp>();

    // Other ops.
    target.addIllegalOp<tosa::AvgPool2dOp,
                        tosa::MaxPool2dOp,
                        tosa::ConcatOp,
                        tosa::Conv2DOp,
                        tosa::DepthwiseConv2DOp,
                        tosa::FullyConnectedOp,
                        tosa::GatherOp,
                        tosa::MatMulOp,
                        tosa::ArgMaxOp,
                        tosa::ReduceAllOp,
                        tosa::ReduceAnyOp,
                        tosa::ReduceMaxOp,
                        tosa::ReduceMinOp,
                        tosa::ReduceProdOp,
                        tosa::ReduceSumOp,
                        tosa::ReshapeOp,
                        tosa::SliceOp,
                        tosa::PadOp,
                        tosa::TileOp,
                        tosa::TransposeOp>();
    // clang-format on

    RewritePatternSet patterns(&getContext());
    populateTosaToEmitcPatterns(&getContext(), patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::emitc::createConvertTosaToEmitCPass() {
  return std::make_unique<ConvertTosaToEmitCPass>();
}
