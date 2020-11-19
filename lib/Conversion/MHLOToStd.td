//===- MHLOToStd.td - Pattern definition file ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the pattern for MHLO const op to std.
//
// Inspired by MLIR-HLO's
// `lib/Dialect/mhlo/transforms/legalize_to_standard_patterns.td`
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"

def : Pat<(HLO_ConstOp ElementsAttr:$value),
          (ConstantOp $value)>;
