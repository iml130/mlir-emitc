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
          DenseIntElementsAttr::get(tensorType, broadcastIndices));

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
  patterns.insert<RsqrtOpConversion>(ctx);
  patterns.insert<CallOpConversion<tosa::TanhOp>>(ctx, "tosa::tanh");

  // Binary elementwise ops
  patterns.insert<CallOpBroadcastableConversion<tosa::AddOp>>(ctx, "tosa::add");
  patterns.insert<MulOpConversion>(ctx, "tosa::mul");
  patterns.insert<CallOpBroadcastableConversion<tosa::SubOp>>(ctx, "tosa::sub");

  // Other ops
  patterns.insert<FullyConnectedOpConversion>(ctx, "tosa::fully_connected");
  patterns.insert<MatMulOpConversion>(ctx);
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
    target.addIllegalOp<tosa::RsqrtOp>();
    target.addIllegalOp<tosa::TanhOp>();

    // Binary elementwise ops
    target.addIllegalOp<tosa::AddOp>();
    target.addIllegalOp<tosa::MulOp>();
    target.addIllegalOp<tosa::SubOp>();

    // Other ops
    target.addIllegalOp<tosa::FullyConnectedOp>();
    target.addIllegalOp<tosa::MatMulOp>();

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
