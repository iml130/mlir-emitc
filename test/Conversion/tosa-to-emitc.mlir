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

func @test_reciprocal(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "tosa::reciprocal"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  // CHECK: emitc.call "tosa::reciprocal"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.reciprocal"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::sqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  // CHECK: %1 = emitc.call "tosa::reciprocal"(%0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

/// Binary elementwise ops

func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  // CHECK: emitc.call "tosa::add"(%0, %arg1) {template_args = []} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// MulOp: no broadcast
func @test_mul10(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::mul"(%arg0, %arg1) {args = [1 : i32], template_args = []} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// MulOp: First operand needs to be broadcasted
func @test_mul1(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  // CHECK: emitc.call "tosa::mul"(%0, %arg1) {args = [1 : i32], template_args = []} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// MulOp: Second operand needs to be broadcasted
func @test_mul2(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  // CHECK: emitc.call "tosa::mul"(%arg0, %0) {args = [1 : i32], template_args = []} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// MulOp: Second operand needs to be broadcasted + expanded
func @test_mul3(%arg0: tensor<21x3xf32>, %arg1: tensor<3xf32>) -> tensor<21x3xf32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [dense<1> : tensor<1xi64>], template_args = [tensor<21x3xf32>]} : (tensor<3xf32>) -> tensor<21x3xf32>
  // CHECK: emitc.call "tosa::mul"(%arg0, %0) {args = [3 : i32], template_args = []} : (tensor<21x3xf32>, tensor<21x3xf32>) -> tensor<21x3xf32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 3 : i32 } : (tensor<21x3xf32>, tensor<3xf32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// MulOp: Second operand needs to be broadcasted + expanded
func @test_mul4(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [dense<2> : tensor<1xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<3xf32>) -> tensor<13x21x3xf32>
  // CHECK: emitc.call "tosa::mul"(%arg0, %0) {args = [1 : i32], template_args = []} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x21x3xf32>, tensor<3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// MulOp: Second operand needs to be broadcasted + expanded with two dimension
func @test_mul5(%arg0: tensor<2x13x21x3xf32>, %arg1: tensor<21x3xf32>) -> tensor<2x13x21x3xf32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [dense<[2, 3]> : tensor<2xi64>], template_args = [tensor<2x13x21x3xf32>]} : (tensor<21x3xf32>) -> tensor<2x13x21x3xf32>
  // CHECK: emitc.call "tosa::mul"(%arg0, %0) {args = [5 : i32], template_args = []} : (tensor<2x13x21x3xf32>, tensor<2x13x21x3xf32>) -> tensor<2x13x21x3xf32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 5 : i32 } : (tensor<2x13x21x3xf32>, tensor<21x3xf32>) -> tensor<2x13x21x3xf32>
  return %0 : tensor<2x13x21x3xf32>
}

/// Other ops
// FullyConnectedOp
func @test_fully_connected(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>, %arg2: tensor<28xf32>) -> tensor<14x28xf32> {
  // CHECK: emitc.call "tosa::fully_connected"(%arg0, %arg1, %arg2) {template_args = []} : (tensor<14x19xf32>, tensor<19x28xf32>, tensor<28xf32>) -> tensor<14x28xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<14x19xf32>, tensor<19x28xf32>, tensor<28xf32>) -> tensor<14x28xf32>
  return %0 : tensor<14x28xf32>
}
