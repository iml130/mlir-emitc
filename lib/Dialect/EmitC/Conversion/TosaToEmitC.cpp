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

/// Common functions.
DenseIntElementsAttr getI64ElementsAttr(const ArrayAttr values,
                                        MLIRContext *ctx) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, IntegerType::get(ctx, 64));

  SmallVector<int64_t> valuesAsI64;

  // Convert values to int64_t.
  for (auto &value : values) {
    auto valueAsIntAttr = value.cast<IntegerAttr>();
    valuesAsI64.push_back(valueAsIntAttr.getInt());
  };

  return DenseIntElementsAttr::get(ty, valuesAsI64);
}

SmallVector<Attribute, 2> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::to_vector<2>(
      llvm::map_range(llvm::seq<int64_t>(0, n), [&ctx](int64_t i) -> Attribute {
        return IntegerAttr::get(IndexType::get(ctx), i);
      }));
}

/// Convert a common `tosa` operation into an `emitc.call` operation.
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

class ConstOpConversion : public OpRewritePattern<tosa::ConstOp> {
public:
  using OpRewritePattern<tosa::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConstOp constOp,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstOp>(constOp, constOp.getType(),
                                                constOp.value());
    return success();
  }
};

/// Convert a common `tosa` convolution operation into an `emitc.call`
/// operation.
template <typename SrcOp>
class GenericConvOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  GenericConvOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp convOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Fail if quantization is requested.
    if (convOp.quantization_info().hasValue()) {
      return convOp.emitError("Quantization for " + convOp.getOperationName() +
                              " is currently not supported.");
    }

    // All tosa conv* ops support adding a bias after the actual convolution. We
    // will split the convolution with bias into two operations: the convolution
    // (as an emitc call op) and the addition of the bias (as an tosa.add op).
    // Therefore we remove bias from the operands of the convolution.
    operands = operands.drop_back();

    StringAttr callee = rewriter.getStringAttr(funcName);

    // clang-format off
    ArrayAttr args = rewriter.getArrayAttr({
      rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(1),
      getI64ElementsAttr(convOp.pad(), convOp.getContext()),
      getI64ElementsAttr(convOp.stride(), convOp.getContext()),
      getI64ElementsAttr(convOp.dilation(), convOp.getContext()),
    });
    // clang-format on

    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(convOp.getResult().getType())});

    // Create conv op.
    auto emitcConvOp =
        rewriter.create<emitc::CallOp>(convOp->getLoc(), convOp.getType(),
                                       callee, args, templateArgs, operands);

    auto output = emitcConvOp.getResult(0);
    auto tosaAddOp = rewriter.create<tosa::AddOp>(
        convOp.getLoc(), output.getType(), output, convOp.bias());

    rewriter.replaceOp(convOp, {tosaAddOp.getResult()});

    return success();
  }

  StringRef funcName;
};

/// Convert `tosa.fully_connected` into an `emitc.call` operation.
class FullyConnectedOpConversion
    : public OpConversionPattern<tosa::FullyConnectedOp> {
  using OpConversionPattern<tosa::FullyConnectedOp>::OpConversionPattern;

public:
  FullyConnectedOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<tosa::FullyConnectedOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::FullyConnectedOp fullyConnectedOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (fullyConnectedOp.quantization_info().hasValue()) {
      return fullyConnectedOp.emitError(
          "Quantization of tosa.fully_connected is currently not supported.");
    }

    StringRef funcName = "tosa::fully_connected";
    StringAttr callee = rewriter.getStringAttr(funcName);

    Type type = fullyConnectedOp.getType();

    ArrayAttr args;
    ArrayAttr templateArgs =
        ArrayAttr::get(fullyConnectedOp.getContext(), {TypeAttr::get(type)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(fullyConnectedOp, type, callee,
                                               args, templateArgs, operands);
    return success();
  }
};

/// Convert `tosa.matmul` into an `emitc.call` operation.
class MatMulOpConversion : public OpConversionPattern<tosa::MatMulOp> {
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;

public:
  MatMulOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::MatMulOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::MatMulOp matMulOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (matMulOp.quantization_info().hasValue()) {
      return matMulOp.emitError(
          "Quantization of tosa.matmul is currently not supported.");
    }

    StringRef funcName = "tosa::matmul";
    StringAttr callee = rewriter.getStringAttr(funcName);

    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        matMulOp, matMulOp.getType(), callee, args, templateArgs, operands);
    return success();
  }
};

/// Convert `tosa.reluN` into an `emitc.call` operation.
class ReluNOpConversion : public OpConversionPattern<tosa::ReluNOp> {
  using OpConversionPattern<tosa::ReluNOp>::OpConversionPattern;

public:
  ReluNOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::ReluNOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::ReluNOp reluNOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef funcName = "tosa::reluN";
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 2> args_;
    args_.push_back(rewriter.getIndexAttr(0));

    // TOSA specifies the max attributes to be either exact i64 or f32,
    // regardless of the operand's element type. So we need to make sure that
    // the max attribute type match the operand's element type and it's bit
    // width.
    auto elementType =
        operands[0].getType().cast<RankedTensorType>().getElementType();
    if (elementType.isa<IntegerType>()) {
      // Change the max_int type to the element type of the operand.
      auto maxInt = reluNOp.max_int();
      args_.push_back(mlir::IntegerAttr::get(elementType, maxInt));
    } else if (elementType.isa<FloatType>()) {
      // Change the max_fp type to the element type of the operand.
      auto maxFp = reluNOp.max_fpAttr().getValueAsDouble();
      args_.push_back(mlir::FloatAttr::get(elementType, maxFp));
    } else {
      return reluNOp.emitError(
          "Operand of tosa.reluN has to be tensor of integer or float values.");
    }
    ArrayAttr args = rewriter.getArrayAttr(args_);
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        reluNOp, reluNOp.getType(), callee, args, templateArgs, operands);

    return success();
  }
};

/// Convert `tosa.rsqrt` into an `emitc.call` operation.
class RsqrtOpConversion : public OpConversionPattern<tosa::RsqrtOp> {
  using OpConversionPattern<tosa::RsqrtOp>::OpConversionPattern;

public:
  RsqrtOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::RsqrtOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::RsqrtOp rsqrtOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayAttr args;
    ArrayAttr templateArgs;

    // Create sqrt op.
    StringRef sqrtFuncName = "emitc::sqrt";
    StringAttr sqrtCallee = rewriter.getStringAttr(sqrtFuncName);

    auto sqrtEmitCOp = rewriter.create<emitc::CallOp>(
        rsqrtOp.getLoc(), rsqrtOp.getType(), sqrtCallee, args, templateArgs,
        operands);

    // Create reciprocal op.
    StringRef reciprocalFuncName = "tosa::reciprocal";
    StringAttr reciprocalCallee = rewriter.getStringAttr(reciprocalFuncName);

    auto reciprocalOp = rewriter.create<emitc::CallOp>(
        sqrtEmitCOp.getLoc(), rsqrtOp.getType(), reciprocalCallee, args,
        templateArgs, sqrtEmitCOp.getResults());

    rewriter.replaceOp(rsqrtOp, reciprocalOp.getResults());

    return success();
  }
};

/// Convert `tosa.fully_connected` into an `emitc.call` operation.
template <typename SrcOp>
SmallVector<Value, 2>
createBroadcastOpIfNeeded(SrcOp &srcOp, ArrayRef<Value> operands,
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
  SmallVector<Value, 2> broadcastedOperands;
  broadcastedOperands.push_back(operands[0]);
  broadcastedOperands.push_back(operands[1]);
  for (size_t i = 0; i < operands.size(); ++i) {
    auto &operand = operands[i];
    auto operandShape = operand.getType().cast<RankedTensorType>().getShape();
    auto operandRank = operand.getType().cast<RankedTensorType>().getRank();

    // Insert a broadcast_in_dim operation if shape of operands don't match.
    if (!operandShape.equals(opOutputShape)) {
      SmallVector<Attribute, 1> broadcastIndices;
      auto numBroadcastDims = opOutputRank - operandRank;
      for (int64_t d = numBroadcastDims; d < opOutputRank; ++d) {
        broadcastIndices.push_back(
            mlir::IntegerAttr::get(rewriter.getIntegerType(64), d));
      }

      RankedTensorType tensorType =
          RankedTensorType::get({static_cast<int64_t>(operandRank)},
                                IntegerType::get(srcOp.getContext(), 64));
      ArrayAttr broadcastArgs = rewriter.getArrayAttr(
          {rewriter.getIndexAttr(0),
           DenseIntElementsAttr::get(tensorType, broadcastIndices)});

      ArrayAttr templateBroadcastArgs =
          rewriter.getArrayAttr({TypeAttr::get(srcOp.getType())});

      auto broadcastArg = rewriter.create<emitc::CallOp>(
          srcOp->getLoc(), srcOp.getType(), broadcastCallee, broadcastArgs,
          templateBroadcastArgs, operand);
      // Replace the original operand with the result of the broadcast_in_dim
      // operation.
      broadcastedOperands[i] = broadcastArg.getResult(0);
    }
  }
  return broadcastedOperands;
}

/// Convert a common, broadcastable `tosa` operation into an `emitc.call`
/// operation.
template <typename SrcOp>
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

    SmallVector<Value, 2> broadcastedOperands =
        createBroadcastOpIfNeeded(srcOp, operands, rewriter);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        srcOp, srcOp.getType(), callee, args, templateArgs,
        mlir::ValueRange({broadcastedOperands[0], broadcastedOperands[1]}));

    return success();
  }

  StringRef funcName;
  // If set, use the result type of the operation as template parameter.
  bool explicitResultType;
  // If set, use the operand types as (additional) template parameters.
  bool explicitOperandTypes;
};

/// Convert `tosa.mul` into an `emitc.call` operation.
class MulOpConversion : public OpConversionPattern<tosa::MulOp> {
  using OpConversionPattern<tosa::MulOp>::OpConversionPattern;

public:
  MulOpConversion(MLIRContext *ctx, StringRef funcName,
                  bool explicitResultType = false,
                  bool explicitOperandTypes = false)
      : OpConversionPattern<tosa::MulOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::MulOp mulOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName = "tosa::mul";
    StringAttr callee = rewriter.getStringAttr(funcName);

    auto shiftAttr = mulOp.shiftAttr();
    ArrayAttr args;
    SmallVector<Attribute, 1> args_;
    if (shiftAttr.getInt() > 0) {
      args_.push_back(rewriter.getIndexAttr(0));
      args_.push_back(rewriter.getIndexAttr(1));
      args_.push_back(shiftAttr);
      args = rewriter.getArrayAttr(args_);
    }

    ArrayAttr templateArgs;

    SmallVector<Value, 2> broadcastedOperands =
        createBroadcastOpIfNeeded(mulOp, operands, rewriter);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        mulOp, mulOp.getType(), callee, args, templateArgs,
        mlir::ValueRange({broadcastedOperands[0], broadcastedOperands[1]}));

    return success();
  }
};

template <typename SrcOp>
class ReduceOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  ReduceOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp reduceOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute> args_ =
        indexSequence(operands.size(), reduceOp.getContext());
    args_.push_back(reduceOp.axisAttr());

    ArrayAttr args = rewriter.getArrayAttr(args_);

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
                                reduceOp.axis());

    auto newOutputType =
        RankedTensorType::get(llvm::makeArrayRef(newReducedOutputShape),
                              reducedOutputType.getElementType());

    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(newOutputType),
                               TypeAttr::get(reduceOp.input().getType())});

    auto emitcReduceOp = rewriter.create<emitc::CallOp>(
        reduceOp.getLoc(), newOutputType, callee, args, templateArgs, operands);

    // Create tosa.reshape op.
    SmallVector<Attribute> newShapeAttr_;
    for (auto dim : output.getType().cast<RankedTensorType>().getShape()) {
      newShapeAttr_.push_back(
          IntegerAttr::get(rewriter.getIntegerType(64), dim));
    };

    ArrayAttr newShapeAttr =
        ArrayAttr::get(reduceOp.getContext(), newShapeAttr_);

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        reduceOp, output.getType(), emitcReduceOp.getResult(0), newShapeAttr);

    return success();
  }

  StringRef funcName;
};

/// Convert `tosa.pad` into an `emitc.call` operation.
class PadOpConversion : public OpConversionPattern<tosa::PadOp> {
  using OpConversionPattern<tosa::PadOp>::OpConversionPattern;

public:
  PadOpConversion(MLIRContext *ctx) : OpConversionPattern<tosa::PadOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::PadOp padOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (padOp.quantization_info().hasValue()) {
      return padOp.emitError(
          "Quantization of tosa.pad is currently not supported.");
    }

    StringAttr callee = rewriter.getStringAttr("tosa::pad");

    // No arguments! Pad itself is an operand and not an argument. Therefore, we
    // have to handle any conversion in tosa::pad.
    ArrayAttr args;

    Type resultType = padOp.output().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(padOp, padOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }
};

} // namespace

void populateTosaToEmitcPatterns(MLIRContext *ctx,
                                 OwningRewritePatternList &patterns) {
  // Insert patterns for TOSA data node ops.
  patterns.insert<ConstOpConversion>(ctx);

  // Insert patterns for TOSA unary elementwise ops.
  patterns.insert<CallOpConversion<tosa::AbsOp>>(ctx, "tosa::abs");
  patterns.insert<CallOpConversion<tosa::CeilOp>>(ctx, "tosa::ceil");
  patterns.insert<CallOpConversion<tosa::ExpOp>>(ctx, "tosa::exp");
  patterns.insert<CallOpConversion<tosa::FloorOp>>(ctx, "tosa::floor");
  patterns.insert<CallOpConversion<tosa::LogOp>>(ctx, "tosa::log");
  patterns.insert<CallOpConversion<tosa::ReciprocalOp>>(ctx,
                                                        "tosa::reciprocal");
  patterns.insert<ReluNOpConversion>(ctx);
  patterns.insert<RsqrtOpConversion>(ctx);
  patterns.insert<CallOpConversion<tosa::TanhOp>>(ctx, "tosa::tanh");

  // Insert patterns for TOSA binary elementwise ops.
  patterns.insert<CallOpBroadcastableConversion<tosa::AddOp>>(ctx, "tosa::add");
  patterns.insert<MulOpConversion>(ctx, "tosa::mul");
  patterns.insert<CallOpBroadcastableConversion<tosa::SubOp>>(ctx, "tosa::sub");

  // Insert patterns for other TOSA ops.
  patterns.insert<GenericConvOpConversion<tosa::Conv2DOp>>(ctx, "tosa::conv2d");
  patterns.insert<GenericConvOpConversion<tosa::DepthwiseConv2DOp>>(
      ctx, "tosa::depthwise_conv2d");
  patterns.insert<FullyConnectedOpConversion>(ctx, "tosa::fully_connected");
  patterns.insert<MatMulOpConversion>(ctx);
  patterns.insert<ReduceOpConversion<tosa::ReduceAllOp>>(ctx,
                                                         "tosa::reduce_all");
  patterns.insert<ReduceOpConversion<tosa::ReduceAnyOp>>(ctx,
                                                         "tosa::reduce_any");
  patterns.insert<ReduceOpConversion<tosa::ReduceMaxOp>>(ctx,
                                                         "tosa::reduce_max");
  patterns.insert<ReduceOpConversion<tosa::ReduceMinOp>>(ctx,
                                                         "tosa::reduce_min");
  patterns.insert<ReduceOpConversion<tosa::ReduceProdOp>>(ctx,
                                                          "tosa::reduce_prod");
  patterns.insert<ReduceOpConversion<tosa::ReduceSumOp>>(ctx,
                                                         "tosa::reduce_sum");
  patterns.insert<CallOpConversion<tosa::ReshapeOp>>(
      ctx, "tosa::reshape", /*explicitResultType=*/true);
  patterns.insert<PadOpConversion>(ctx);
  patterns.insert<CallOpConversion<tosa::TransposeOp>>(
      ctx, "tosa::transpose", /*explicitResultType=*/true);
}

namespace {

struct ConvertTosaToEmitCPass
    : public ConvertTosaToEmitCBase<ConvertTosaToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<tosa::TosaDialect>();
    // TODO: We might to readd further legal dialects and ops
    // target.addLegalDialect<StandardOpsDialect>();
    // target.addLegalOp<FuncOp>();
    // target.addLegalOp<ModuleOp>();
    // target.addLegalOp<ModuleTerminatorOp>();

    // Data node ops
    target.addIllegalOp<tosa::ConstOp>();

    // Unary elementwise ops
    target.addIllegalOp<tosa::AbsOp>();
    target.addIllegalOp<tosa::CeilOp>();
    target.addIllegalOp<tosa::ExpOp>();
    target.addIllegalOp<tosa::FloorOp>();
    target.addIllegalOp<tosa::LogOp>();
    target.addIllegalOp<tosa::ReciprocalOp>();
    target.addIllegalOp<tosa::ReluNOp>();
    target.addIllegalOp<tosa::RsqrtOp>();
    target.addIllegalOp<tosa::TanhOp>();

    // Binary elementwise ops
    target.addIllegalOp<tosa::AddOp>();
    target.addIllegalOp<tosa::MulOp>();
    target.addIllegalOp<tosa::SubOp>();

    // Other ops
    target.addIllegalOp<tosa::Conv2DOp>();
    target.addIllegalOp<tosa::DepthwiseConv2DOp>();
    target.addIllegalOp<tosa::FullyConnectedOp>();
    target.addIllegalOp<tosa::MatMulOp>();
    target.addIllegalOp<tosa::ReduceAllOp>();
    target.addIllegalOp<tosa::ReduceAnyOp>();
    target.addIllegalOp<tosa::ReduceMaxOp>();
    target.addIllegalOp<tosa::ReduceMinOp>();
    target.addIllegalOp<tosa::ReduceProdOp>();
    target.addIllegalOp<tosa::ReduceSumOp>();
    target.addIllegalOp<tosa::ReshapeOp>();
    target.addIllegalOp<tosa::PadOp>();
    target.addIllegalOp<tosa::TransposeOp>();

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
