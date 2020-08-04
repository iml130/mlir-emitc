// RUN: emitc-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: emitc-opt %s | emitc-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: emitc-opt -mlir-print-op-generic %s | emitc-opt | FileCheck %s

func @std_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  emitc.for %i0 = %arg0 to %arg1 step %arg2 {
    emitc.for %i1 = %arg0 to %arg1 step %arg2 {
      %min_cmp = cmpi "slt", %i0, %i1 : index
      %min = select %min_cmp, %i0, %i1 : index
      %max_cmp = cmpi "sge", %i0, %i1 : index
      %max = select %max_cmp, %i0, %i1 : index
      emitc.for %i2 = %min to %max step %i1 {
      }
    }
  }
  return
}
// CHECK-LABEL: func @std_for(
//  CHECK-NEXT:   emitc.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:     emitc.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:       %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       emitc.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {

func @std_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = constant 0.0 : f32
  %result = emitc.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (f32) {
    %sn = addf %si, %si : f32
    emitc.yield %sn : f32
  }
  return
}
// CHECK-LABEL: func @std_for_yield(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]:
// CHECK-NEXT: %[[INIT:.*]] = constant
// CHECK-NEXT: %{{.*}} = emitc.for %{{.*}} = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
// CHECK-SAME: iter_args(%[[ITER:.*]] = %[[INIT]]) -> (f32) {
// CHECK-NEXT: %[[NEXT:.*]] = addf %[[ITER]], %[[ITER]] : f32
// CHECK-NEXT: emitc.yield %[[NEXT]] : f32
// CHECK-NEXT: }


func @std_for_yield_multi(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = constant 0.0 : f32
  %t0 = constant 1 : i32
  %u0 = constant 1.0 : f32
  %result1:3 = emitc.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0, %ti = %t0, %ui = %u0) -> (f32, i32, f32) {
    %sn = addf %si, %si : f32
    %tn = addi %ti, %ti : i32
    %un = subf %ui, %ui : f32
    emitc.yield %sn, %tn, %un : f32, i32, f32
  }
  return
}
// CHECK-LABEL: func @std_for_yield_multi(
// CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]:
// CHECK-NEXT: %[[INIT1:.*]] = constant
// CHECK-NEXT: %[[INIT2:.*]] = constant
// CHECK-NEXT: %[[INIT3:.*]] = constant
// CHECK-NEXT: %{{.*}}:3 = emitc.for %{{.*}} = %[[ARG0]] to %[[ARG1]] step %[[ARG2]]
// CHECK-SAME: iter_args(%[[ITER1:.*]] = %[[INIT1]], %[[ITER2:.*]] = %[[INIT2]], %[[ITER3:.*]] = %[[INIT3]]) -> (f32, i32, f32) {
// CHECK-NEXT: %[[NEXT1:.*]] = addf %[[ITER1]], %[[ITER1]] : f32
// CHECK-NEXT: %[[NEXT2:.*]] = addi %[[ITER2]], %[[ITER2]] : i32
// CHECK-NEXT: %[[NEXT3:.*]] = subf %[[ITER3]], %[[ITER3]] : f32
// CHECK-NEXT: emitc.yield %[[NEXT1]], %[[NEXT2]], %[[NEXT3]] : f32, i32, f32
