//===- ConvertToScf.cpp - Convert MHLO Whileto SCF ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "convert-to-scf"

namespace mlir {
namespace mhlo {

namespace {

void ConvertWhile(WhileOp while_op);

class ConvertToScfPass
    : public mlir::PassWrapper<ConvertToScfPass, FunctionPass> {
  void runOnFunction() override {
    getFunction().walk([&](Operation *op) {
      if (auto while_op = llvm::dyn_cast<WhileOp>(op)) {
        ConvertWhile(while_op);
      }
    });
  }
};

void ConvertWhile(WhileOp while_op) {
  // Handle pattern;
  //   x = start
  //   step = ...
  //   limit = ...
  //   while (x < limit) { ... x += step; }

  // Only handling multi value while loops at the moment;
  auto tupleOp = while_op.getOperand().getDefiningOp<TupleOp>();
  if (!tupleOp)
    return;
  auto bodyReturn = while_op.body()
                        .front()
                        .getTerminator()
                        ->getOperand(0)
                        .getDefiningOp<mhlo::TupleOp>();
  assert(bodyReturn && "invalid mhlo::While");

  auto result = while_op.cond().front().getTerminator()->getOperand(0);
  auto cmp = result.getDefiningOp<mhlo::CompareOp>();
  if (!cmp || cmp.comparison_direction() != "LT")
    return;

  const int kConstant = -1;
  auto getThrough = [&](Value val) -> std::pair<Value, int> {
    if (matchPattern(val, m_Constant()))
      return {val, kConstant};
    if (auto gte = val.getDefiningOp<GetTupleElementOp>()) {
      if (!gte.getOperand().isa<mlir::BlockArgument>())
        return {nullptr, 0};
      int index = gte.index().getSExtValue();
      return {tupleOp.getOperand(index), index};
    }
    return {nullptr, 0};
  };

  std::pair<Value, int> min, max, step;
  min = getThrough(cmp.lhs());
  max = getThrough(cmp.rhs());
  if (!min.first || !max.first)
    return;
  auto add = bodyReturn.getOperand(min.second).getDefiningOp<mhlo::AddOp>();
  if (!add)
    return;
  if (!matchPattern(add.rhs(), m_Constant()))
    return;
  step = getThrough(add.rhs());
  if (step.second != kConstant)
    return;

  if (!min.first || !max.first || !step.first)
    return;

  // Only handle case where tuple isn't propagated as is for now.
  for (auto *use : while_op.body().front().getArgument(0).getUsers()) {
    if (!isa<GetTupleElementOp>(use))
      return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Found if:\n";
             llvm::dbgs() << "min = " << min.second << " max = " << max.second
                          << " step = " << step.second << "\n";
             llvm::dbgs() << "min = " << min.first << " max = " << max.first
                          << " step = " << step.first << "\n";);
  OpBuilder b(while_op);
  // Inputs to new for loop.
  llvm::SmallVector<Value, 4> input;
  input.reserve(tupleOp.getNumOperands());
  for (auto r : tupleOp.getOperands().take_front(min.second))
    input.push_back(r);
  for (auto r : tupleOp.getOperands().drop_front(min.second + 1))
    input.push_back(r);

  auto tensorIndexType = RankedTensorType::get({}, b.getIndexType());
  auto getAsIndex = [&](Value val) {
    auto loc = NameLoc::get(b.getIdentifier("extract"), while_op.getLoc());
    return b.create<ExtractElementOp>(
        loc, b.create<IndexCastOp>(loc, tensorIndexType, val), ValueRange{});
  };

  auto forMin = getAsIndex(min.first);
  auto forMax = getAsIndex(max.first);
  auto forStep = getAsIndex(step.first);
  auto forOp = b.create<mlir::scf::ForOp>(while_op.getLoc(), forMin, forMax,
                                          forStep, input);
  forOp.getLoopBody().front().getOperations().splice(
      forOp.getLoopBody().front().getOperations().end(),
      while_op.body().front().getOperations());

  b.setInsertionPointToStart(&forOp.getLoopBody().front());
  auto minElType = min.first.getType().cast<ShapedType>().getElementType();
  Value indVar;
  {
    // Short term hack with custom op to work around lack of support in std.
    auto loc = NameLoc::get(b.getIdentifier("create"), while_op.getLoc());
    OperationState state(loc, "foo.get_as_int");
    state.addOperands(forOp.getInductionVar());
    state.addTypes(RankedTensorType::get({}, minElType));
    indVar = b.createOperation(state)->getResult(0);
  }
  for (auto *use :
       llvm::make_early_inc_range(while_op.body().getArgument(0).getUsers())) {
    auto gte = cast<GetTupleElementOp>(use);
    if (gte.index() == min.second) {
      use->getResult(0).replaceAllUsesWith(indVar);
      use->erase();
      continue;
    }
    int index = gte.index().getSExtValue();
    if (index > min.second)
      --index;
    use->getResult(0).replaceAllUsesWith(forOp.getIterOperands()[index]);
    use->erase();
  }

  SmallVector<Value, 4> newYieldOps;
  newYieldOps.reserve(bodyReturn.getNumOperands() - 1);
  for (auto r : bodyReturn.getOperands().take_front(min.second))
    newYieldOps.push_back(r);
  for (auto r : bodyReturn.getOperands().drop_front(min.second + 1))
    newYieldOps.push_back(r);

  // Delete return & tuple op.
  forOp.getLoopBody().front().getOperations().back().erase();
  forOp.getLoopBody().front().getOperations().back().erase();
  b.setInsertionPointToEnd(&forOp.getLoopBody().front());
  b.create<scf::YieldOp>(while_op.getLoc(), newYieldOps);

  // Recombine output tuple with max value of induction variable.
  llvm::SmallVector<Value, 4> loopOut;
  loopOut.reserve(forOp.getNumResults() + 1);
  for (auto r : forOp.getResults().take_front(min.second))
    loopOut.push_back(r);
  loopOut.push_back(max.first);
  for (auto r : forOp.getResults().drop_front(min.second))
    loopOut.push_back(r);
  b.setInsertionPoint(while_op.getOperation());
  auto newRes = b.create<mhlo::TupleOp>(while_op.getLoc(), loopOut);
  while_op.replaceAllUsesWith(newRes.getOperation());

  while_op.erase();
  cast<scf::YieldOp>(forOp.getLoopBody().front().getTerminator());
}

} // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertToScfPass() {
  return std::make_unique<ConvertToScfPass>();
}

} // namespace mhlo
} // namespace mlir
