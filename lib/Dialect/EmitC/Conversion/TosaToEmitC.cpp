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
create_broadcast_op_if_needed(SrcOp &srcOp, ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter) {
  // TOSA allows implicit broadcasting, so we need to insert broadcast_in_dim
  // Ops if necessary
  // clang-format off
  // tosa.add tensor<8xf32>)     -> tensor<1x4x4x8xf32>     Broadcast dims = (3)
  // tosa.add tensor<4x8xf32>)   -> tensor<1x4x4x8xf32>     Broadcast dims = (2, 3) 
  // tosa.add tensor<4x4x8xf32>) -> tensor<1x4x4x8xf32>     Broadcast dims = (1, 2, 3)
  // clang-format on

  StringRef broadcastFuncName = "emitc::broadcast_in_dim";
  StringAttr broadcastCallee = rewriter.getStringAttr(broadcastFuncName);

  Value output = srcOp.getResult();
  auto opOutputShape = output.getType().cast<RankedTensorType>().getShape();
  auto opOutputRank = output.getType().cast<RankedTensorType>().getRank();
  SmallVector<Value, 2> broadcastedOperands;
  broadcastedOperands.push_back(operands[0]);
  broadcastedOperands.push_back(operands[1]);
  for (size_t i = 0; i < operands.size(); i++) {
    auto &operand = operands[i];
    auto operandShape = operand.getType().cast<RankedTensorType>().getShape();
    auto operandRank = operand.getType().cast<RankedTensorType>().getRank();

    // Insert a broadcast_in_dim Operation if shape of operands don't match
    if (!operandShape.equals(opOutputShape)) {
      SmallVector<Attribute, 1> broadcastIndices;
      auto numBroadcastDims = opOutputRank - operandRank;
      for (int64_t i = numBroadcastDims; i < opOutputRank; i++) {
        broadcastIndices.push_back(
            mlir::IntegerAttr::get(rewriter.getIntegerType(64), i));
      }

      RankedTensorType ty =
          RankedTensorType::get({static_cast<int64_t>(operandRank)},
                                IntegerType::get(srcOp.getContext(), 64));
      ArrayAttr broadcastArgs = rewriter.getArrayAttr(
          DenseIntElementsAttr::get(ty, broadcastIndices));

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
        create_broadcast_op_if_needed(srcOp, operands, rewriter);

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
      : OpConversionPattern<tosa::MulOp>(ctx), funcName(funcName),
        explicitResultType(explicitResultType),
        explicitOperandTypes(explicitOperandTypes) {}

private:
  LogicalResult
  matchAndRewrite(tosa::MulOp mulOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr(funcName);

    SmallVector<Attribute, 4> templateArgs_;

    if (explicitResultType) {
      Type type = mulOp.getType();
      templateArgs_.push_back(TypeAttr::get(type));
    }

    if (explicitOperandTypes) {
      for (auto &operand : operands) {
        Type type = operand.getType();
        templateArgs_.push_back(TypeAttr::get(type));
      }
    }

    SmallVector<Attribute, 1> args_;
    args_.push_back(mulOp.shiftAttr());
    ArrayAttr args = rewriter.getArrayAttr(args_);

    ArrayAttr templateArgs = ArrayAttr::get(mulOp.getContext(), templateArgs_);

    SmallVector<Value, 2> broadcastedOperands =
        create_broadcast_op_if_needed(mulOp, operands, rewriter);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        mulOp, mulOp.getType(), callee, args, templateArgs,
        mlir::ValueRange({broadcastedOperands[0], broadcastedOperands[1]}));
    
    return success();
  }

  StringRef funcName;
  // If set, use the result type of the operation as template parameter
  bool explicitResultType;
  // If set, use the operand types as (additional) template parameters
  bool explicitOperandTypes;
};

} // namespace

void populateTosaToEmitcPatterns(MLIRContext *ctx,
                                 OwningRewritePatternList &patterns) {
  // Unary elementwise ops
  patterns.insert<CallOpConversion<tosa::AbsOp>>(ctx, "tosa::abs");
  patterns.insert<CallOpConversion<tosa::ExpOp>>(ctx, "tosa::exp");
  patterns.insert<CallOpConversion<tosa::ReciprocalOp>>(ctx,
                                                        "tosa::reciprocal");
  patterns.insert<RsqrtOpConversion>(ctx);

  // Binary elementwise ops
  patterns.insert<CallOpConversion<tosa::AddOp>>(ctx, "tosa::add");
  patterns.insert<CallOpConversion<tosa::MulOp>>(ctx, "tosa::mul");

  patterns.insert<CallOpBroadcastableConversion<tosa::AddOp>>(ctx, "tosa::add");
  patterns.insert<MulOpConversion>(ctx, "tosa::mul");

  // Other ops
  patterns.insert<CallOpConversion<tosa::FullyConnectedOp>>(
      ctx, "tosa::fully_connected");
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

    // Unary elementwise ops
    target.addIllegalOp<tosa::AbsOp>();
    target.addIllegalOp<tosa::ExpOp>();
    target.addIllegalOp<tosa::ReciprocalOp>();
    target.addIllegalOp<tosa::RsqrtOp>();

    // Binary elementwise ops
    target.addIllegalOp<tosa::AddOp>();
    target.addIllegalOp<tosa::MulOp>();

    // Other ops
    target.addIllegalOp<tosa::FullyConnectedOp>();

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
