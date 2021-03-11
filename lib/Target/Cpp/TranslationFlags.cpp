//===- TranslationFlags.cpp - EmitC translation flags -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "emitc/Target/TranslationFlags.h"
#include "emitc/Target/Cpp.h"

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace emitc {

static llvm::cl::opt<bool> forwardDeclareFlag{
    "forward-declare-variables",
    llvm::cl::desc("Forward declare all variables emitted from emitc"),
    llvm::cl::init(true),
};

TargetOptions getTargetOptionsFromFlags() {
  TargetOptions targetOptions;
  targetOptions.forwardDeclareVariables = forwardDeclareFlag;
  return targetOptions;
}

} // namespace emitc
} // namespace mlir
