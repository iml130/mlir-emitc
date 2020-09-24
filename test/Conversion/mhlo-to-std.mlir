// RUN: emitc-opt -convert-mhlo-const-to-std %s | emitc-translate --mlir-to-cpp | FileCheck %s

// CHECK: Tensor1D<int32_t, 2> mhlo_constant(Tensor1D<int32_t, 2> v1)
func @mhlo_constant(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "mhlo.constant"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: return v2;
  return %0 : tensor<2xi32>
}
