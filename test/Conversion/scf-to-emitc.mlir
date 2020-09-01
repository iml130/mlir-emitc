// RUN: emitc-opt --convert-scf-to-emitc %s | FileCheck %s

func @std_if(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
    %0 = addf %arg1, %arg1 : f32
  }
  return
}
// CHECK-LABEL: func @std_if(
//  CHECK-NEXT:   emitc.if %{{.*}} {
//  CHECK-NEXT:     %{{.*}} = addf %{{.*}}, %{{.*}} : f32

func @std_if_else(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
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

func @std_if_yield(%arg0: i1, %arg1: f32) -> (f32, f32)
{
  %x, %y = scf.if %arg0 -> (f32, f32) {
    %0 = addf %arg1, %arg1 : f32
    %1 = subf %arg1, %arg1 : f32
    scf.yield %0, %1 : f32, f32
  } else {
    %0 = subf %arg1, %arg1 : f32
    %1 = addf %arg1, %arg1 : f32
    scf.yield %0, %1 : f32, f32
  }
  return %x, %y : f32, f32
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


func @std_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    scf.for %i1 = %arg0 to %arg1 step %arg2 {
      %min_cmp = cmpi "slt", %i0, %i1 : index
      %min = select %min_cmp, %i0, %i1 : index
      %max_cmp = cmpi "sge", %i0, %i1 : index
      %max = select %max_cmp, %i0, %i1 : index
      scf.for %i2 = %min to %max step %i1 {
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

func @std_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) -> f32 {
  %s0 = constant 0.0 : f32
  %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (f32) {
    %sn = addf %si, %si : f32
    scf.yield %sn : f32
  }
  return %result : f32
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


func @std_for_yield_multi(%arg0 : index, %arg1 : index, %arg2 : index) -> i32 {
  %s0 = constant 0.0 : f32
  %t0 = constant 1 : i32
  %u0 = constant 1.0 : f32
  %result1:3 = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0, %ti = %t0, %ui = %u0) -> (f32, i32, f32) {
    %sn = addf %si, %si : f32
    %tn = addi %ti, %ti : i32
    %un = subf %ui, %ui : f32
    scf.yield %sn, %tn, %un : f32, i32, f32
  }
  return  %result1#1 : i32
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
