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

// Common functions
DenseIntElementsAttr getI64ElementsAttr(const ArrayAttr values,
                                        MLIRContext *ctx) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, IntegerType::get(ctx, 64));

  SmallVector<int64_t> valuesAsI64;

  // convert values to int64_t
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

    ArrayAttr templateArgs = ArrayAttr::get(srcOp.getContext(), templateArgs_);

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

class Conv2DOpConversion : public OpConversionPattern<tosa::Conv2DOp> {
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;

public:
  Conv2DOpConversion(MLIRContext *ctx)
      : OpConversionPattern<tosa::Conv2DOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(tosa::Conv2DOp conv2dOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // fail if quantization is requested
    if (conv2dOp.quantization_info().hasValue()) {
      return conv2dOp.emitError(
          "Quantization of tosa.conv2d is currently not supported.");
    }

    // tosa conv2D supports adding a bias after the actual convolution. We will
    // split the convolution with bias into two operations: the convolution (as
    // an emitc call op) and the addition of the bias (as an tosa.add op).
    // Therefore we remove bias from the operands of the convolution.
    operands = operands.drop_back();

    StringRef funcName = "tosa::conv2d";
    StringAttr callee = rewriter.getStringAttr(funcName);

    // clang-format off
    ArrayAttr args = rewriter.getArrayAttr({
      rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(1),
      getI64ElementsAttr(conv2dOp.pad(), conv2dOp.getContext()),
      getI64ElementsAttr(conv2dOp.stride(), conv2dOp.getContext()),
      getI64ElementsAttr(conv2dOp.dilation(), conv2dOp.getContext()),
    });
    // clang-format on

    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(conv2dOp.getResult().getType())});

    // create conv2dOp
    auto emitcConvOp =
        rewriter.create<emitc::CallOp>(conv2dOp->getLoc(), conv2dOp.getType(),
                                       callee, args, templateArgs, operands);

    auto output = emitcConvOp.getResult(0);
    auto tosaAddOp = rewriter.create<tosa::AddOp>(
        conv2dOp.getLoc(), output.getType(), output, conv2dOp.bias());

    rewriter.replaceOp(conv2dOp, {tosaAddOp.getResult()});

    return success();
  }
};

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

    // Since tosa.reluN has two max attribute types for float and integer
    // values, we have to determine to which max attribute we have to clamp to
    auto elementType =
        operands[0].getType().cast<RankedTensorType>().getElementType();
    if (elementType.isa<IntegerType>()) {
      args_.push_back(reluNOp.max_intAttr());
    } else if (elementType.isa<FloatType>()) {
      args_.push_back(reluNOp.max_fpAttr());
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

    // create sqrt op
    StringRef sqrtFuncName = "emitc::sqrt";
    StringAttr sqrtCallee = rewriter.getStringAttr(sqrtFuncName);

    auto sqrtEmitCOp = rewriter.create<emitc::CallOp>(
        rsqrtOp.getLoc(), rsqrtOp.getType(), sqrtCallee, args, templateArgs,
        operands);

    // create reciprocal op
    StringRef reciprocalFuncName = "tosa::reciprocal";
    StringAttr reciprocalCallee = rewriter.getStringAttr(reciprocalFuncName);

    auto reciprocalOp = rewriter.create<emitc::CallOp>(
        sqrtEmitCOp.getLoc(), rsqrtOp.getType(), reciprocalCallee, args,
        templateArgs, sqrtEmitCOp.getResults());

    rewriter.replaceOp(rsqrtOp, reciprocalOp.getResults());

    return success();
  }
};

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

    // Insert a broadcast_in_dim Operation if shape of operands don't match
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
      // operation
      broadcastedOperands[i] = broadcastArg.getResult(0);
    }
  }
  return broadcastedOperands;
}

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

    ArrayAttr templateArgs = ArrayAttr::get(srcOp.getContext(), templateArgs_);

    SmallVector<Value, 2> broadcastedOperands =
        createBroadcastOpIfNeeded(srcOp, operands, rewriter);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        srcOp, srcOp.getType(), callee, args, templateArgs,
        mlir::ValueRange({broadcastedOperands[0], broadcastedOperands[1]}));

    return success();
  }

  StringRef funcName;
  // If set, use the result type of the operation as template parameter
  bool explicitResultType;
  // If set, use the operand types as (additional) template parameters
  bool explicitOperandTypes;
};

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

    // we need to adjust output shape of reduce since our implementation does
    // not keep reduced dimensions
    Value output = reduceOp.getResult();
    RankedTensorType reducedOutputType =
        output.getType().cast<RankedTensorType>();

    SmallVector<int64_t> newReducedOutputShape;

    for (auto dim : reducedOutputType.getShape()) {
      newReducedOutputShape.push_back(dim);
    };

    // remove reduced axis from shape
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

    // create tosa.reshape op
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

} // namespace

void populateTosaToEmitcPatterns(MLIRContext *ctx,
                                 OwningRewritePatternList &patterns) {
  // Data node ops
  patterns.insert<ConstOpConversion>(ctx);

  // Unary elementwise ops
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

  // Binary elementwise ops
  patterns.insert<CallOpBroadcastableConversion<tosa::AddOp>>(ctx, "tosa::add");
  patterns.insert<MulOpConversion>(ctx, "tosa::mul");
  patterns.insert<CallOpBroadcastableConversion<tosa::SubOp>>(ctx, "tosa::sub");

  // Other ops
  patterns.insert<Conv2DOpConversion>(ctx);
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
  patterns.insert<CallOpConversion<tosa::TransposeOp>>(
      ctx, "tosa::transpose", /*explicitResultType=*/true);
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
    target.addIllegalOp<tosa::FullyConnectedOp>();
    target.addIllegalOp<tosa::MatMulOp>();
    target.addIllegalOp<tosa::ReduceAllOp>();
    target.addIllegalOp<tosa::ReduceAnyOp>();
    target.addIllegalOp<tosa::ReduceMaxOp>();
    target.addIllegalOp<tosa::ReduceMinOp>();
    target.addIllegalOp<tosa::ReduceProdOp>();
    target.addIllegalOp<tosa::ReduceSumOp>();
    target.addIllegalOp<tosa::ReshapeOp>();
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
