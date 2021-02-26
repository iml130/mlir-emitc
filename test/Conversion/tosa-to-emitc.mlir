// RUN: emitc-opt -convert-tosa-to-emitc %s | FileCheck %s

/// Unary elementwise ops

func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::abs"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::exp"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::sqrt"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  // CHECK: %1 = emitc.call "tosa::reciprocal"(%0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

/// Binary elementwise ops

func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::add"(%arg0, %arg1) {template_args = []} : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}
