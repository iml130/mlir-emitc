//===- Pipelines.cpp - EmitC Pipeline Passes ------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "emitc/Conversion/Passes.h"
#include "emitc/Dialect/EmitC/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace emitc {
namespace {

#ifdef EMITC_BUILD_HLO
void buildStablehloToEmitCPipeline(OpPassManager &pm) {
  pm.addPass(createInsertEmitCStablehloIncludePass());
  pm.addPass(createConvertStablehloRegionOpsToEmitCPass());
  pm.addPass(createConvertStablehloToEmitCPass());
}
#endif // EMITC_BUILD_HLO

void buildArithToEmitCPipeline(OpPassManager &pm) {
  pm.addPass(createInsertEmitCArithIncludePass());
  pm.addPass(createConvertArithToEmitCPass());
}

void buildTensorToEmitCPipeline(OpPassManager &pm) {
  pm.addPass(createInsertEmitCTensorIncludePass());
  pm.addPass(createConvertTensorToEmitCPass());
}

void buildTosaToEmitCPipeline(OpPassManager &pm) {
  pm.addPass(createInsertEmitCTosaIncludePass());
  pm.addPass(createConvertTosaToEmitCPass());
}

} // namespace

#ifdef EMITC_BUILD_HLO
void registerStablehloToEmitCPipeline() {
  PassPipelineRegistration<>("stablehlo-to-emitc-pipeline",
                             "Run the StableHLO to EmitC pipeline.",
                             buildStablehloToEmitCPipeline);
}
#endif // EMITC_BUILD_HLO

void registerArithToEmitCPipeline() {
  PassPipelineRegistration<>("arith-to-emitc-pipeline",
                             "Run the Arithmetic to EmitC pipeline.",
                             buildArithToEmitCPipeline);
}

void registerTensorToEmitCPipeline() {
  PassPipelineRegistration<>("tensor-to-emitc-pipeline",
                             "Run the Tensor to EmitC pipeline.",
                             buildTensorToEmitCPipeline);
}

void registerTosaToEmitCPipeline() {
  PassPipelineRegistration<>("tosa-to-emitc-pipeline",
                             "Run the TOSA to EmitC pipeline.",
                             buildTosaToEmitCPipeline);
}

} // namespace emitc
} // namespace mlir
