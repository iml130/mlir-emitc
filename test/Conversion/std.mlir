// RUN: emitc-opt -convert-std-to-emitc %s | emitc-translate --mlir-to-cpp | FileCheck %s

// CHECK: std::vector<size_t> std_index_cast(std::vector<size_t> v1, std::vector<int32_t> v2)
func @std_index_cast(%arg0: tensor<index>, %arg1: tensor<2xi32>) -> tensor<2xindex> {
  // CHECK: standard::index_cast<int32_t>(v1)
  %0 = "std.index_cast"(%arg0) : ( tensor<index>) -> tensor<i32>
  // CHECK: standard::index_cast<size_t>(v2)
  %1 = "std.index_cast"(%arg1) : (tensor<2xi32>) -> tensor<2xindex>
  // CHECK: return v4;
  return %1 : tensor<2xindex>
}
