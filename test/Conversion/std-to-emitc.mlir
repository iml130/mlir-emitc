// RUN: emitc-opt -convert-std-to-emitc %s | FileCheck %s
// RUN: emitc-opt -convert-std-to-emitc %s | emitc-translate --mlir-to-cpp | FileCheck %s -check-prefix=CPP
// RUN: emitc-opt -insert-emitc-std-include -convert-std-to-emitc %s | FileCheck %s --check-prefixes=CHECK,CHECK-INCLUDE
// RUN: emitc-opt -std-to-emitc-pipeline %s | FileCheck %s --check-prefixes=CHECK,CHECK-INCLUDE

// CHECK-INCLUDE: emitc.include "emitc_std.h"

func @splat_op(%s : f32) -> tensor<8xf32> {
  %t = splat %s : tensor<8xf32>
  return %t : tensor<8xf32>
}
// CHECK-LABEL: func @splat_op
//  CHECK-NEXT: emitc.call "emitc::standard::splat"(%arg0) {template_args = [tensor<8xf32>]} : (f32) -> tensor<8xf32>

// CPP-LABEL: Tensor<float, 8> splat_op(float v1)
//  CPP-NEXT: emitc::standard::splat<Tensor<float, 8>>(v1)
//  CPP-NEXT: return v2;
