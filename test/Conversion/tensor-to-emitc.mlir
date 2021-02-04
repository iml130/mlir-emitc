// RUN: emitc-opt -convert-tensor-to-emitc %s | emitc-translate --mlir-to-cpp | FileCheck %s

// CHECK: void std_extract_element(Tensor<int32_t> v1, Tensor<int32_t, 2> v2)
func @std_extract_element(%arg0: tensor<i32>, %arg1: tensor<2xi32>) -> () {
  %0 = constant 0 : index
  %1 = constant 1 : index
  // CHECK: tensor::extract(v1)
  %2 = tensor.extract %arg0[] : tensor<i32>
  // CHECK: tensor::extract(v2, v3)
  %3 = tensor.extract %arg1[%0] : tensor<2xi32>
  // CHECK: tensor::extract(v2, v4)
  %4 = tensor.extract %arg1[%1] : tensor<2xi32>
  return 
}
