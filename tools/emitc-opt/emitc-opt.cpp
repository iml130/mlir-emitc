//===- emitc-opt.cpp - MLIR Optimizer Driver ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for emitc-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "emitc/InitDialect.h"
#include "emitc/InitPasses.h"
#ifdef EMITC_BUILD_HLO
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#endif
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registerAllPasses();
  registerEmitCDialect(registry);
  emitc::registerAllEmitCPasses();
#ifdef EMITC_BUILD_HLO
  registry.insert<mlir::mhlo::MhloDialect>();
  mlir::mhlo::registerLegalizeControlFlowToScfPassPass();
#endif // EMITC_BUILD_HLO

  return failed(MlirOptMain(argc, argv, "MLIR EmitC modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
