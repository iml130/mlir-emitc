// RUN: emitc-opt -convert-std-to-emitc %s | emitc-translate --mlir-to-cpp | FileCheck %s

// CHECK: void std_extract_element(Tensor0D<int32_t> v1, Tensor1D<int32_t, 2> v2)
func @std_extract_element(%arg0: tensor<i32>, %arg1: tensor<2xi32>) -> () {
  %0 = constant 0 : index
  %1 = constant 1 : index
  // CHECK: standard::extract_element(v1)
  %2 = extract_element %arg0[] : tensor<i32>
  // CHECK: standard::extract_element(v2, v3)
  %3 = extract_element %arg1[%0] : tensor<2xi32>
  // CHECK: standard::extract_element(v2, v4)
  %4 = extract_element %arg1[%1] : tensor<2xi32>
  return 
}

// CHECK: Tensor1D<size_t, 2> std_index_cast(Tensor0D<size_t> v1, Tensor1D<int32_t, 2> v2, Tensor2D<int32_t, 2, 2> v3)
func @std_index_cast(%arg0: tensor<index>, %arg1: tensor<2xi32>, %arg2: tensor<2x2xi32>) -> tensor<2xindex> {
  // CHECK: standard::index_cast<Tensor0D<int32_t>>(v1)
  %0 = "std.index_cast"(%arg0) : ( tensor<index>) -> tensor<i32>
  // CHECK: standard::index_cast<Tensor1D<size_t, 2>>(v2)
  %1 = "std.index_cast"(%arg1) : (tensor<2xi32>) -> tensor<2xindex>
  // CHECK: standard::index_cast<Tensor2D<size_t, 2, 2>>(v3)
  %2 = "std.index_cast"(%arg2) : (tensor<2x2xi32>) -> tensor<2x2xindex>
  // CHECK: return v5;
  return %1 : tensor<2xindex>
}

// CHECK: Tensor1D<float, 8> splat_op(float v1)
func @splat_op(%s : f32) -> tensor<8xf32> {
  // CHECK: standard::splat<Tensor1D<float, 8>>(v1)
  %t = splat %s : tensor<8xf32>
  return %t : tensor<8xf32>
}
