//===- MHLORegionOpsToEmitC.cpp - MHLO to EmitC conversion ----------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for converting MHLO ops containing regions to the
// EmitC dialect by outlining the regions to module level functions.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "emitc/Dialect/EmitC/Conversion/Passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

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

struct ConvertMhloRegionOpsToEmitCPass
    : public ConvertMHLORegionOpsToEmitCBase<ConvertMhloRegionOpsToEmitCPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<EmitCDialect, func::FuncDialect>();
  }

  /// Perform the lowering to EmitC dialect.
  void runOnOperation() override {
    // Convert region ops
    SymbolTable symbolTable(getOperation());
    for (auto func : getOperation().getOps<FuncOp>()) {
      // Insert just before the function.
      Block::iterator insertPt(func.getOperation());

      int count = 0;
      // ReduceOp
      auto funcWalkResult = func.walk([&](mhlo::ReduceOp op) {
        std::string funcName =
            Twine(op->getParentOfType<FuncOp>().getName(), "_lambda_")
                .concat(Twine(count++))
                .str();

        Optional<FuncOp> outlinedFunc =
            outlineRegionImpl<mhlo::ReduceOp>(op, funcName);

        if (!outlinedFunc.hasValue()) {
          return WalkResult::interrupt();
        }

        symbolTable.insert(outlinedFunc.getValue(), insertPt);

        if (failed(convertToCall(op, outlinedFunc.getValue()))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (funcWalkResult.wasInterrupted())
        return signalPassFailure();

      // ReduceWindowOp
      funcWalkResult = func.walk([&](mhlo::ReduceWindowOp op) {
        std::string funcName =
            Twine(op->getParentOfType<FuncOp>().getName(), "_lambda_")
                .concat(Twine(count++))
                .str();

        Optional<FuncOp> outlinedFunc =
            outlineRegionImpl<mhlo::ReduceWindowOp>(op, funcName);

        if (!outlinedFunc.hasValue()) {
          return WalkResult::interrupt();
        }

        symbolTable.insert(outlinedFunc.getValue(), insertPt);

        if (failed(convertToCall(op, outlinedFunc.getValue()))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (funcWalkResult.wasInterrupted())
        return signalPassFailure();
    }
  }

private:
  template <typename OpType>
  Optional<FuncOp> outlineRegionImpl(OpType &op, std::string functionName) {
    Location loc = op.getLoc();
    // Create a builder with no insertion point, insertion will happen
    // separately due to symbol table manipulation.
    OpBuilder builder(op.getContext());

    Region &region = op.getRegion();

    auto &blocks = region.getBlocks();

    if (blocks.size() > 1) {
      return None;
    }

    Block &block = blocks.front();
    auto *terminator = block.getTerminator();
    auto returnOp = dyn_cast_or_null<mhlo::ReturnOp>(terminator);
    if (!returnOp) {
      return None;
    }

    auto inputs = region.getArgumentTypes();
    auto results = returnOp.getOperandTypes();

    FunctionType type = FunctionType::get(op.getContext(), inputs, results);
    auto outlinedFunc = builder.create<FuncOp>(loc, functionName, type);

    Region &outlinedRegion = outlinedFunc.getRegion();

    BlockAndValueMapping map;
    region.cloneInto(&outlinedRegion, map);

    outlinedFunc.walk([](mhlo::ReturnOp returnOp) {
      OpBuilder replacer(returnOp);
      replacer.create<func::ReturnOp>(returnOp.getLoc(),
                                      returnOp.getOperands());
      returnOp.erase();
    });
    return outlinedFunc;
  }

  LogicalResult convertToCall(mhlo::ReduceOp &op, FuncOp &funcOp) {
    OpBuilder builder(op);
    auto *ctx = op.getContext();

    auto operands = op.getOperands();

    StringRef funcName = "emitc::mhlo::reduce";
    StringAttr callee = StringAttr::get(ctx, funcName);

    SmallVector<Attribute, 2> args_ =
        indexSequence(operands.size(), op.getContext());

    args_.push_back(op.dimensions());
    args_.push_back(SymbolRefAttr::get(ctx, funcOp.getName()));

    ArrayAttr args = ArrayAttr::get(ctx, args_);

    SmallVector<Attribute, 2> templateArgs_ = llvm::to_vector<2>(
        llvm::map_range(llvm::seq<size_t>(0, op.getNumResults()),
                        [&op](size_t i) -> Attribute {
                          return TypeAttr::get(op.getResults()[i].getType());
                        }));

    templateArgs_.push_back(
        IntegerAttr::get(IntegerType::get(ctx, 64), op.dimensions().size()));

    ArrayAttr templateArgs = ArrayAttr::get(ctx, templateArgs_);

    emitc::CallOp callOp = builder.create<emitc::CallOp>(
        op.getLoc(), op.getResultTypes(), callee, args, templateArgs, operands);
    op.replaceAllUsesWith(callOp);
    op.erase();
    return success();
  }

  LogicalResult convertToCall(mhlo::ReduceWindowOp &op, FuncOp &funcOp) {
    OpBuilder builder(op);
    auto *ctx = op.getContext();

    auto operands = op.getOperands();

    StringRef funcName = "emitc::mhlo::reduce_window";
    StringAttr callee = StringAttr::get(ctx, funcName);

    SmallVector<Attribute, 2> args_ = indexSequence(operands.size(), ctx);

    size_t dim = op.getResult(0).getType().cast<RankedTensorType>().getRank();
    args_.push_back(op.window_dimensions());
    args_.push_back(
        op.window_strides().getValueOr(i64ElementsAttr(1, dim, ctx)));
    args_.push_back(
        op.base_dilations().getValueOr(i64ElementsAttr(1, dim, ctx)));
    args_.push_back(
        op.window_dilations().getValueOr(i64ElementsAttr(1, dim, ctx)));
    args_.push_back(op.padding().getValueOr(i64ElementsAttr(0, 2 * dim, ctx)));
    args_.push_back(SymbolRefAttr::get(ctx, funcOp.getName()));

    ArrayAttr args = ArrayAttr::get(ctx, args_);

    ArrayAttr templateArgs =
        ArrayAttr::get(ctx, {TypeAttr::get(op.getResult(0).getType())});

    emitc::CallOp callOp = builder.create<emitc::CallOp>(
        op.getLoc(), op.getType(0), callee, args, templateArgs, operands);
    op.replaceAllUsesWith(callOp);
    op.erase();
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::emitc::createConvertMhloRegionOpsToEmitCPass() {
  return std::make_unique<ConvertMhloRegionOpsToEmitCPass>();
}
