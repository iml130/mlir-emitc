// RUN: emitc-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: emitc-opt %s | emitc-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: emitc-opt -mlir-print-op-generic %s | emitc-opt | FileCheck %s

func @std_if(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
    %0 = addf %arg1, %arg1 : f32
  }
  return
}
// CHECK-LABEL: func @std_if(
//  CHECK-NEXT:   emitc.if %{{.*}} {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32

func @std_if_else(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
    %0 = addf %arg1, %arg1 : f32
  } else {
    %1 = addf %arg1, %arg1 : f32
  }
  return
}
// CHECK-LABEL: func @std_if_else(
//  CHECK-NEXT:   emitc.if %{{.*}} {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32

func @std_if_yield(%arg0: i1, %arg1: f32)
{
  %x, %y = emitc.if %arg0 -> (f32, f32) {
    %0 = addf %arg1, %arg1 : f32
    %1 = subf %arg1, %arg1 : f32
    emitc.yield %0, %1 : f32, f32
  } else {
    %0 = subf %arg1, %arg1 : f32
    %1 = addf %arg1, %arg1 : f32
    emitc.yield %0, %1 : f32, f32
  }
  return
}
// CHECK-LABEL: func @std_if_yield(
//  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
//  CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
//  CHECK-NEXT: %{{.*}}:2 = emitc.if %[[ARG0]] -> (f32, f32) {
//  CHECK-NEXT: %[[T1:.*]] = addf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: %[[T2:.*]] = subf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: emitc.yield %[[T1]], %[[T2]] : f32, f32
//  CHECK-NEXT: } else {
//  CHECK-NEXT: %[[T3:.*]] = subf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: %[[T4:.*]] = addf %[[ARG1]], %[[ARG1]]
//  CHECK-NEXT: emitc.yield %[[T3]], %[[T4]] : f32, f32
//  CHECK-NEXT: }
