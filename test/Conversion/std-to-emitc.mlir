// RUN: emitc-opt -convert-std-to-emitc %s | emitc-translate --mlir-to-cpp | FileCheck %s

// CHECK: Tensor<size_t, 2> std_index_cast(Tensor<size_t> v1, Tensor<int32_t, 2> v2, Tensor<int32_t, 2, 2> v3)
func @std_index_cast(%arg0: tensor<index>, %arg1: tensor<2xi32>, %arg2: tensor<2x2xi32>) -> tensor<2xindex> {
  // CHECK: standard::index_cast<Tensor<int32_t>>(v1)
  %0 = "std.index_cast"(%arg0) : ( tensor<index>) -> tensor<i32>
  // CHECK: standard::index_cast<Tensor<size_t, 2>>(v2)
  %1 = "std.index_cast"(%arg1) : (tensor<2xi32>) -> tensor<2xindex>
  // CHECK: standard::index_cast<Tensor<size_t, 2, 2>>(v3)
  %2 = "std.index_cast"(%arg2) : (tensor<2x2xi32>) -> tensor<2x2xindex>
  // CHECK: return v5;
  return %1 : tensor<2xindex>
}

// CHECK: Tensor<float, 8> splat_op(float v1)
func @splat_op(%s : f32) -> tensor<8xf32> {
  // CHECK: standard::splat<Tensor<float, 8>>(v1)
  %t = splat %s : tensor<8xf32>
  return %t : tensor<8xf32>
}
