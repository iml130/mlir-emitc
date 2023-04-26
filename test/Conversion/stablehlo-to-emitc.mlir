// RUN: emitc-opt -convert-stablehlo-to-emitc %s | FileCheck %s

// Unary elementwise ops

func.func @stablehlo_abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call "emitc::mhlo::abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
