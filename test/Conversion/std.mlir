// RUN: emitc-opt -convert-std-to-emitc %s | emitc-translate --mlir-to-cpp | FileCheck %s

// CHECK: void std_extract_element(std::vector<int32_t> v1, std::vector<int32_t> v2)
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

// CHECK: std::vector<size_t> std_index_cast(std::vector<size_t> v1, std::vector<int32_t> v2)
func @std_index_cast(%arg0: tensor<index>, %arg1: tensor<2xi32>) -> tensor<2xindex> {
  // CHECK: standard::index_cast<int32_t>(v1)
  %0 = "std.index_cast"(%arg0) : ( tensor<index>) -> tensor<i32>
  // CHECK: standard::index_cast<size_t>(v2)
  %1 = "std.index_cast"(%arg1) : (tensor<2xi32>) -> tensor<2xindex>
  // CHECK: return v4;
  return %1 : tensor<2xindex>
}
