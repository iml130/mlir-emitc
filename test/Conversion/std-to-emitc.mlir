// RUN: emitc-opt -convert-std-to-emitc %s | FileCheck %s
// RUN: emitc-opt -convert-std-to-emitc %s | emitc-translate --mlir-to-cpp | FileCheck %s -check-prefix=CPP

func @std_index_cast(%arg0: tensor<index>, %arg1: tensor<2xi32>, %arg2: tensor<2x2xi32>) -> tensor<2xindex> {
  %0 = "std.index_cast"(%arg0) : ( tensor<index>) -> tensor<i32>
  %1 = "std.index_cast"(%arg1) : (tensor<2xi32>) -> tensor<2xindex>
  %2 = "std.index_cast"(%arg2) : (tensor<2x2xi32>) -> tensor<2x2xindex>
  return %1 : tensor<2xindex>
}
// CHECK-LABEL: func @std_index_cast
//  CHECK-NEXT: emitc.call "emitc::standard::index_cast"(%arg0) {template_args = [tensor<i32>]} : (tensor<index>) -> tensor<i32>
//  CHECK-NEXT: emitc.call "emitc::standard::index_cast"(%arg1) {template_args = [tensor<2xindex>]} : (tensor<2xi32>) -> tensor<2xindex>
//  CHECK-NEXT: emitc.call "emitc::standard::index_cast"(%arg2) {template_args = [tensor<2x2xindex>]} : (tensor<2x2xi32>) -> tensor<2x2xindex>

// CPP-LABEL: Tensor<size_t, 2> std_index_cast(Tensor<size_t> v1, Tensor<int32_t, 2> v2, Tensor<int32_t, 2, 2> v3)
//  CPP-NEXT: emitc::standard::index_cast<Tensor<int32_t>>(v1)
//  CPP-NEXT: emitc::standard::index_cast<Tensor<size_t, 2>>(v2)
//  CPP-NEXT: emitc::standard::index_cast<Tensor<size_t, 2, 2>>(v3)
//  CPP-NEXT: return v5;

func @splat_op(%s : f32) -> tensor<8xf32> {
  %t = splat %s : tensor<8xf32>
  return %t : tensor<8xf32>
}
// CHECK-LABEL: func @splat_op
//  CHECK-NEXT: emitc.call "emitc::standard::splat"(%arg0) {template_args = [tensor<8xf32>]} : (f32) -> tensor<8xf32>

// CPP-LABEL: Tensor<float, 8> splat_op(float v1)
//  CPP-NEXT: emitc::standard::splat<Tensor<float, 8>>(v1)
//  CPP-NEXT: return v2;
