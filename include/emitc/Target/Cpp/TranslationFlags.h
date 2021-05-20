//===- TranslationFlags.h - EmitC translation flags -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_TARGET_CPP_TRANSLATIONFLAGS_H
#define EMITC_TARGET_CPP_TRANSLATIONFLAGS_H

#include "emitc/Target/Cpp/Cpp.h"

namespace mlir {
namespace emitc {

// Returns a TargetOptions struct initialized with the command line flags.
TargetOptions getTargetOptionsFromFlags();

} // namespace emitc
} // namespace mlir

#endif // EMITC_TARGET_CPP_TRANSLATIONFLAGS_H
