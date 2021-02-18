// RUN: emitc-opt -convert-tosa-to-emitc %s | FileCheck %s

func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}
