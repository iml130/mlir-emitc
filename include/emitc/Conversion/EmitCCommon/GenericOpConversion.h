//===- GenericOpConversion.cpp - Op to EmitC call op conversion -----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_CONVERSION_EMITCCOMMON_GENERICOPCONVERSION_H
#define EMITC_CONVERSION_EMITCCOMMON_GENERICOPCONVERSION_H

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace mlir::emitc;

/// Convert a common operation into an `emitc.call_opaque` operation.
template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class GenericOpConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  GenericOpConversion(MLIRContext *ctx, StringRef funcName,
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

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(srcOp, srcOp.getType(),
                                                     callee, args, templateArgs,
                                                     adaptor.getOperands());

    return success();
  }

  StringRef funcName;
  // If set, use the result type of the operation as template parameter.
  bool explicitResultType;
  // If set, use the operand types as (additional) template parameters.
  bool explicitOperandTypes;
};

} // namespace

#endif // EMITC_CONVERSION_EMITCCOMMON_GENERICOPCONVERSION_H
