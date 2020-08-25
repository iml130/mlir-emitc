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

class BroadcastInDimOpConversion
    : public OpConversionPattern<mhlo::BroadcastInDimOp> {

public:
  BroadcastInDimOpConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::BroadcastInDimOp broadcastInDimOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: Generalize to other cases
    auto broadcastDims = broadcastInDimOp.broadcast_dimensions();
    if (broadcastDims.getType().getRank() != 1)
      return broadcastInDimOp.emitError()
             << "broadcast_dimensions with rank other than 1 not supported";
    if (broadcastDims.getType().getShape().front() != 0)
      return broadcastInDimOp.emitError()
             << "broadcast_dimensions with size other than 0 not supported";
    if (auto result = broadcastInDimOp.getResult()
                          .getType()
                          .dyn_cast<RankedTensorType>()) {
      if (result.getRank() != 1)
        return failure();

      auto size = result.getShape().front();

      StringRef funcName = "mhlo::broadcast_in_dim";
      StringAttr callee = rewriter.getStringAttr(funcName);
      ArrayAttr args;
      ArrayAttr templateArgs = rewriter.getArrayAttr(
          {IntegerAttr::get(rewriter.getIntegerType(32), size)});

      rewriter.replaceOpWithNewOp<emitc::CallOp>(
          broadcastInDimOp, broadcastInDimOp.getType(), callee, args,
          templateArgs, operands);

      return success();
    }

    return failure();
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

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        srcOp, srcOp.getType(), callee, ArrayAttr{}, templateArgs, operands);

    return success();
  }
};

template <typename SrcOp, typename DstOp>
class UnaryOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  UnaryOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename SrcOp::Adaptor srcAdapter(operands);

    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args;
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<DstOp>(srcOp, srcOp.getType(), callee, args,
                                       templateArgs, operands);

    return success();
  }

  StringRef funcName;
};

// Adopted from IREE's ConvertStandardToVM/ConvertVMToEmitC.
template <typename SrcOp, typename DstOp>
class BinaryOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  BinaryOpConversion(MLIRContext *ctx, StringRef funcName)
      : OpConversionPattern<SrcOp>(ctx), funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename SrcOp::Adaptor srcAdapter(operands);

    StringAttr callee = rewriter.getStringAttr(funcName);
    ArrayAttr args;
    ArrayAttr templateArgs;
    ValueRange dstOperands{srcAdapter.lhs(), srcAdapter.rhs()};

    rewriter.replaceOpWithNewOp<DstOp>(srcOp, srcAdapter.lhs().getType(),
                                       callee, args, templateArgs, dstOperands);

    return success();
  }

  StringRef funcName;
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
    llvm::StringRef functionName;
    if (comparisonDirection.equals("EQ"))
      functionName = "std::equal_to";
    else if (comparisonDirection.equals("NE"))
      functionName = "std::not_equal_to";
    else if (comparisonDirection.equals("GE"))
      functionName = "std::greater_equal";
    else if (comparisonDirection.equals("GT"))
      functionName = "std::greater";
    else if (comparisonDirection.equals("LE"))
      functionName = "std::less_equal";
    else if (comparisonDirection.equals("LT"))
      functionName = "std::less";
    else
      return failure();

    Type elementType = compareOp.getOperand(0).getType();
    if (auto tensorType = elementType.dyn_cast<TensorType>()) {
      elementType = tensorType.getElementType();
    }
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
    auto index = getTupleElementOp.index().getZExtValue();

    // TODO Consider adding template arguments to CallOp
    StringAttr callee = rewriter.getStringAttr("std::get");

    ArrayAttr args;
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {IntegerAttr::get(rewriter.getIntegerType(32), index)});
    ;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        getTupleElementOp, getTupleElementOp.getType(), callee, args,
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

    Type elementType = bitcastConvertOp.getType();
    if (auto tensorType = elementType.dyn_cast<TensorType>()) {
      elementType = tensorType.getElementType();
    }
    ArrayAttr args;
    ArrayAttr templateArgs =
        rewriter.getArrayAttr({TypeAttr::get(elementType)});
    ;

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
    ;

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

class RngUniformOpConversion : public OpConversionPattern<mhlo::RngUniformOp> {
  using OpConversionPattern<mhlo::RngUniformOp>::OpConversionPattern;

public:
  RngUniformOpConversion(MLIRContext *ctx)
      : OpConversionPattern<mhlo::RngUniformOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(mhlo::RngUniformOp rngUniformOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename mhlo::RngUniformOp::Adaptor srcAdapter(operands);

    StringAttr callee = rewriter.getStringAttr("mhlo::rng_uniform");

    Type elementType = rngUniformOp.getType();
    if (auto tensorType = elementType.dyn_cast<TensorType>()) {
      elementType = tensorType.getElementType();
    }
    if (auto result =
            rngUniformOp.getResult().getType().dyn_cast<RankedTensorType>()) {
      ArrayAttr args;
      ArrayAttr templateArgs =
          rewriter.getArrayAttr({TypeAttr::get(elementType)});
      ;

      rewriter.replaceOpWithNewOp<emitc::CallOp>(rngUniformOp,
                                                 rngUniformOp.getType(), callee,
                                                 args, templateArgs, operands);

      return success();
    }

    return failure();
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
  patterns.insert<UnaryOpConversion<mhlo::AbsOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::abs");
  patterns.insert<UnaryOpConversion<mhlo::CosOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::cos");
  patterns.insert<UnaryOpConversion<mhlo::ExpOp, emitc::CallOp>>(
      ctx, "mhlo::exponential");
  patterns.insert<UnaryOpConversion<mhlo::IsFiniteOp, emitc::CallOp>>(
      ctx, "mhlo::isfinite");
  patterns.insert<UnaryOpConversion<mhlo::LogOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::log");
  patterns.insert<UnaryOpConversion<mhlo::NegOp, emitc::CallOp>>(
      ctx, "mhlo::negate");
  patterns.insert<UnaryOpConversion<mhlo::SinOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::sin");
  patterns.insert<UnaryOpConversion<mhlo::SqrtOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::sqrt");

  /// Insert patterns for MHLO binary elementwise ops.
  patterns.insert<BinaryOpConversion<mhlo::AddOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::add");
  patterns.insert<BinaryOpConversion<mhlo::DivOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::div");
  patterns.insert<BinaryOpConversion<mhlo::MaxOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::max");
  patterns.insert<BinaryOpConversion<mhlo::MinOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::min");
  patterns.insert<BinaryOpConversion<mhlo::MulOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::mul");
  patterns.insert<BinaryOpConversion<mhlo::PowOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::pow");
  patterns.insert<BinaryOpConversion<mhlo::ShiftLeftOp, emitc::CallOp>>(
      ctx, "mhlo::shift_left");
  patterns.insert<BinaryOpConversion<mhlo::ShiftRightLogicalOp, emitc::CallOp>>(
      ctx, "mhlo::shift_right_logical");
  patterns.insert<BinaryOpConversion<mhlo::SubOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::sub");

  // Insert patterns for MHLO MHLO binary logical elementwise ops.
  patterns.insert<BinaryOpConversion<mhlo::OrOp, emitc::CallOp>>(ctx,
                                                                 "mhlo::or");
  patterns.insert<BinaryOpConversion<mhlo::XorOp, emitc::CallOp>>(ctx,
                                                                  "mhlo::xor");

  // Insert patterns for MHLO tuple ops.
  patterns.insert<CompareOpConversion>(ctx);
  patterns.insert<TupleOpConversion>(ctx);
  patterns.insert<GetTupleElementOpConversion>(ctx);

  // Insert patterns for MHLO slice ops.
  // TODO:
  //  mhlo::SliceOp
  //  mhlo::DynamicSliceOp
  //  mhlo::DynamicUpdateSliceOp

  // Insert patterns for other MHLO ops.
  patterns.insert<BitcastConvertOpConversion>(ctx);
  patterns.insert<BroadcastInDimOpConversion>(ctx);
  patterns.insert<ConvertOpConversion>(ctx);
  patterns.insert<ConcatenateOpConversion>(ctx);
  patterns.insert<ReshapeOpConversion>(ctx);
  patterns.insert<SelectOpConversion>(ctx);

  // Insert patterns for MHLO RNG ops.
  patterns.insert<RngUniformOpConversion>(ctx);
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
    target.addIllegalOp<mhlo::AbsOp, mhlo::CosOp, mhlo::ExpOp, mhlo::IsFiniteOp,
                        mhlo::LogOp, mhlo::NegOp, mhlo::SinOp, mhlo::SqrtOp>();
    target.addIllegalOp<mhlo::AddOp, mhlo::DivOp, mhlo::MaxOp, mhlo::MinOp,
                        mhlo::MulOp, mhlo::PowOp, mhlo::ShiftLeftOp,
                        mhlo::ShiftRightLogicalOp, mhlo::SubOp>();
    target.addIllegalOp<mhlo::OrOp, mhlo::XorOp>();
    target.addIllegalOp<mhlo::BitcastConvertOp, mhlo::BroadcastInDimOp,
                        mhlo::ConvertOp, mhlo::ConcatenateOp, mhlo::ReshapeOp,
                        mhlo::SelectOp>();
    target.addIllegalOp<mhlo::CompareOp, mhlo::TupleOp,
                        mhlo::GetTupleElementOp>();
    target.addIllegalOp<mhlo::RngUniformOp, mhlo::RngBitGeneratorOp>();

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
