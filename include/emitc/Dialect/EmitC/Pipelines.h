//===- Pieplines.h - EmitC Pipeline Passes ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines pipeline passes owned by the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_DIALECT_EMITC_PIPELINES_H
#define EMITC_DIALECT_EMITC_PIPELINES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace emitc {

#ifdef EMITC_BUILD_HLO
void registerStablehloToEmitCPipeline();
#endif // EMITC_BUILD_HLO
void registerArithToEmitCPipeline();
void registerTensorToEmitCPipeline();
void registerTosaToEmitCPipeline();

} // namespace emitc
} // namespace mlir

#endif // EMITC_DIALECT_EMITC_PIPELINES_H
