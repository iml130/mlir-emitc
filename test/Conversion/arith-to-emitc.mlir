// RUN: emitc-opt -convert-arith-to-emitc-ext %s | FileCheck %s
// RUN: emitc-opt -convert-arith-to-emitc-ext %s | emitc-translate --mlir-to-cpp | FileCheck %s -check-prefix=CPP
// RUN: emitc-opt -insert-emitc-arith-include -convert-arith-to-emitc-ext %s | FileCheck %s --check-prefixes=CHECK,CHECK-INCLUDE
// RUN: emitc-opt -arith-to-emitc-pipeline %s | FileCheck %s --check-prefixes=CHECK,CHECK-INCLUDE

// CHECK-INCLUDE: emitc.include "emitc/arith.h"

func.func @arith_index_cast(%arg0: tensor<index>, %arg1: tensor<2xi32>, %arg2: tensor<2x2xi32>) -> tensor<2xindex> {
  %0 = "arith.index_cast"(%arg0) : ( tensor<index>) -> tensor<i32>
  %1 = "arith.index_cast"(%arg1) : (tensor<2xi32>) -> tensor<2xindex>
  %2 = "arith.index_cast"(%arg2) : (tensor<2x2xi32>) -> tensor<2x2xindex>
  return %1 : tensor<2xindex>
}
// CHECK-LABEL: func @arith_index_cast
//  CHECK-NEXT: emitc.call_opaque "emitc::arith::index_cast"(%arg0) {template_args = [tensor<i32>]} : (tensor<index>) -> tensor<i32>
//  CHECK-NEXT: emitc.call_opaque "emitc::arith::index_cast"(%arg1) {template_args = [tensor<2xindex>]} : (tensor<2xi32>) -> tensor<2xindex>
//  CHECK-NEXT: emitc.call_opaque "emitc::arith::index_cast"(%arg2) {template_args = [tensor<2x2xindex>]} : (tensor<2x2xi32>) -> tensor<2x2xindex>

// CPP-LABEL: Tensor<size_t, 2> arith_index_cast(Tensor<size_t> v1, Tensor<int32_t, 2> v2, Tensor<int32_t, 2, 2> v3)
//  CPP-NEXT: emitc::arith::index_cast<Tensor<int32_t>>(v1)
//  CPP-NEXT: emitc::arith::index_cast<Tensor<size_t, 2>>(v2)
//  CPP-NEXT: emitc::arith::index_cast<Tensor<size_t, 2, 2>>(v3)
//  CPP-NEXT: return v5;
