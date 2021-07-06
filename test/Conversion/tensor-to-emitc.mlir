// RUN: emitc-opt -convert-tensor-to-emitc %s | FileCheck %s
// RUN: emitc-opt -convert-tensor-to-emitc %s | emitc-translate --mlir-to-cpp | FileCheck %s -check-prefix=CPP
// RUN: emitc-opt -insert-emitc-tensor-include -convert-tensor-to-emitc %s | FileCheck %s --check-prefixes=CHECK,CHECK-INCLUDE
// RUN: emitc-opt -tensor-to-emitc-pipeline %s | FileCheck %s --check-prefixes=CHECK,CHECK-INCLUDE

// CHECK-INCLUDE: emitc.include "emitc_tensor.h"

func @std_extract_element(%arg0: tensor<i32>, %arg1: tensor<2xi32>) -> () {
  %0 = constant 0 : index
  %1 = constant 1 : index
  %2 = tensor.extract %arg0[] : tensor<i32>
  %3 = tensor.extract %arg1[%0] : tensor<2xi32>
  %4 = tensor.extract %arg1[%1] : tensor<2xi32>
  return 
}
// CHECK-LABEL: func @std_extract_element
//  CHECK-NEXT: constant 0 : index
//  CHECK-NEXT: constant 1 : index
//  CHECK-NEXT: emitc.call "emitc::tensor::extract"(%arg0) : (tensor<i32>) -> i32
//  CHECK-NEXT: emitc.call "emitc::tensor::extract"(%arg1, %c0) : (tensor<2xi32>, index) -> i32
//  CHECK-NEXT: emitc.call "emitc::tensor::extract"(%arg1, %c1) : (tensor<2xi32>, index) -> i32

// CPP-LABEL: void std_extract_element(Tensor<int32_t> v1, Tensor<int32_t, 2> v2)
//  CPP-NEXT: size_t v3 = 0;
//  CPP-NEXT: size_t v4 = 1;
//  CPP-NEXT: emitc::tensor::extract(v1)
//  CPP-NEXT: emitc::tensor::extract(v2, v3)
//  CPP-NEXT: emitc::tensor::extract(v2, v4)
