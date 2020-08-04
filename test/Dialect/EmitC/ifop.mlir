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
