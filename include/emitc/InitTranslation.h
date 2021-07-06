//===- InitTranslations.h - EmitC Translation Registration-------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of translations
// in and out of MLIR to the system.
//
//===----------------------------------------------------------------------===//

#ifndef EMITC_INITTRANSLATION_H
#define EMITC_INITTRANSLATION_H

namespace mlir {

void registerToCppTranslation();

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerEmitCTranslation() {
  static bool init_once = []() {
    registerToCppTranslation();
    return true;
  }();
  (void)init_once;
}
} // namespace mlir

#endif // EMITC_INITTRANSLATION_H
