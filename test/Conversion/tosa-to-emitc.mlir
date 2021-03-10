// RUN: emitc-opt -convert-tosa-to-emitc %s | FileCheck %s

// Data node ops

func @test_const(%arg0 : index) -> tensor<4xi32> {
    // CHECK: "emitc.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %0 = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    return %0 : tensor<4xi32>
}


// Unary elementwise ops

func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::abs"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::ceil"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.ceil"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::exp"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::floor"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.floor"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::log"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.log"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_reciprocal(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "tosa::reciprocal"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.reciprocal"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::sqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  // CHECK: %1 = emitc.call "tosa::reciprocal"(%0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_tanh(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "tosa::tanh"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.tanh"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// Binary elementwise ops

func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  // CHECK: emitc.call "tosa::add"(%0, %arg1) {template_args = []} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// MulOp: no broadcast
func @test_mul10(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "tosa::mul"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 0 : i32 } : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// MulOp: First operand needs to be broadcasted
func @test_mul1(%arg0: tensor<13x1x3xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xi32>]} : (tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  // CHECK: emitc.call "tosa::mul"(%0, %arg1) {args = [0 : index, 1 : index, 1 : i32]} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x1x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// MulOp: Second operand needs to be broadcasted
func @test_mul2(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xi32>]} : (tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  // CHECK: emitc.call "tosa::mul"(%arg0, %0) {args = [0 : index, 1 : index, 1 : i32]} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// MulOp: Second operand needs to be broadcasted + expanded to two dimensions
func @test_mul3(%arg0: tensor<21x3xi32>, %arg1: tensor<3xi32>) -> tensor<21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [dense<1> : tensor<1xi64>], template_args = [tensor<21x3xi32>]} : (tensor<3xi32>) -> tensor<21x3xi32>
  // CHECK: emitc.call "tosa::mul"(%arg0, %0) {args = [0 : index, 1 : index, 3 : i32]} : (tensor<21x3xi32>, tensor<21x3xi32>) -> tensor<21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 3 : i32 } : (tensor<21x3xi32>, tensor<3xi32>) -> tensor<21x3xi32>
  return %0 : tensor<21x3xi32>
}

// MulOp: Second operand needs to be broadcasted + expanded to three dimensions
func @test_mul4(%arg0: tensor<13x21x3xi32>, %arg1: tensor<3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [dense<2> : tensor<1xi64>], template_args = [tensor<13x21x3xi32>]} : (tensor<3xi32>) -> tensor<13x21x3xi32>
  // CHECK: emitc.call "tosa::mul"(%arg0, %0) {args = [0 : index, 1 : index, 1 : i32]} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x21x3xi32>, tensor<3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// MulOp: Second two dimensional operand needs to be broadcasted + expanded to four dimensions
func @test_mul5(%arg0: tensor<2x13x21x3xi32>, %arg1: tensor<21x3xi32>) -> tensor<2x13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [dense<[2, 3]> : tensor<2xi64>], template_args = [tensor<2x13x21x3xi32>]} : (tensor<21x3xi32>) -> tensor<2x13x21x3xi32>
  // CHECK: emitc.call "tosa::mul"(%arg0, %0) {args = [0 : index, 1 : index, 5 : i32]} : (tensor<2x13x21x3xi32>, tensor<2x13x21x3xi32>) -> tensor<2x13x21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 5 : i32 } : (tensor<2x13x21x3xi32>, tensor<21x3xi32>) -> tensor<2x13x21x3xi32>
  return %0 : tensor<2x13x21x3xi32>
}

func @test_sub(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  // CHECK: emitc.call "tosa::sub"(%0, %arg1) {template_args = []} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// Other ops

func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
    // CHECK: %0 = emitc.call "tosa::conv2d"(%arg0, %arg1) {args = [0 : index, 1 : index, dense<0> : tensor<4xi64>, dense<1> : tensor<2xi64>, dense<1> : tensor<2xi64>], template_args = [tensor<1x4x4x8xf32>]} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32>
    // CHECK: %1 = emitc.call "emitc::broadcast_in_dim"(%arg2) {args = [dense<3> : tensor<1xi64>], template_args = [tensor<1x4x4x8xf32>]} : (tensor<8xf32>) -> tensor<1x4x4x8xf32>
    // CHECK: %2 = emitc.call "tosa::add"(%0, %1) {template_args = []} : (tensor<1x4x4x8xf32>, tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>
    return %0 : tensor<1x4x4x8xf32>
}

func @test_fully_connected(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>, %arg2: tensor<28xf32>) -> tensor<14x28xf32> {
  // CHECK: emitc.call "tosa::fully_connected"(%arg0, %arg1, %arg2) {template_args = [tensor<14x28xf32>]} : (tensor<14x19xf32>, tensor<19x28xf32>, tensor<28xf32>) -> tensor<14x28xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<14x19xf32>, tensor<19x28xf32>, tensor<28xf32>) -> tensor<14x28xf32>
  return %0 : tensor<14x28xf32>
}

func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
  // CHECK: emitc.call "tosa::matmul"(%arg0, %arg1) : (tensor<14x19xf32>, tensor<19x28xf32>) -> tensor<14x28xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<14x19xf32>, tensor<19x28xf32>) -> tensor<14x28xf32>
  return %0 : tensor<14x28xf32>
}

// Reduce ops
func @test_reduce_all(%arg0: tensor<13x21x3xi1>) -> tensor<1x21x3xi1> {
  // CHECK: %0 = emitc.call "tosa::reduce_all"(%arg0) {args = [0 : index, 0], template_args = [tensor<21x3xi1>, tensor<13x21x3xi1>]} : (tensor<13x21x3xi1>) -> tensor<21x3xi1>
  // CHECK: %1 = emitc.call "tosa::reshape"(%0) {template_args = [tensor<1x21x3xi1>]} : (tensor<21x3xi1>) -> tensor<1x21x3xi1>
  %0 = "tosa.reduce_all"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<13x1x3xi1> {
  // CHECK: %0 = emitc.call "tosa::reduce_any"(%arg0) {args = [0 : index, 1], template_args = [tensor<13x3xi1>, tensor<13x21x3xi1>]} : (tensor<13x21x3xi1>) -> tensor<13x3xi1>
  // %1 = emitc.call "tosa::reshape"(%0) {template_args = [tensor<13x1x3xi1>]} : (tensor<13x3xi1>) -> tensor<13x1x3xi1>
  %0 = "tosa.reduce_any"(%arg0) {axis = 1 : i64} : (tensor<13x21x3xi1>) -> tensor<13x1x3xi1>
  return %0 : tensor<13x1x3xi1>
}

func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<1x21x3xf32> {
  // CHECK: %0 = emitc.call "tosa::reduce_max"(%arg0) {args = [0 : index, 0], template_args = [tensor<21x3xf32>, tensor<13x21x3xf32>]} : (tensor<13x21x3xf32>) -> tensor<21x3xf32>
  // CHECK: %1 = emitc.call "tosa::reshape"(%0) {template_args = [tensor<1x21x3xf32>]} : (tensor<21x3xf32>) -> tensor<1x21x3xf32>
  %0 = "tosa.reduce_max"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  return %0 : tensor<1x21x3xf32>
}

func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<13x1x3xf32> {
  // CHECK: %0 = emitc.call "tosa::reduce_min"(%arg0) {args = [0 : index, 1], template_args = [tensor<13x3xf32>, tensor<13x21x3xf32>]} : (tensor<13x21x3xf32>) -> tensor<13x3xf32>
  // CHECK: %1 = emitc.call "tosa::reshape"(%0) {template_args = [tensor<13x1x3xf32>]} : (tensor<13x3xf32>) -> tensor<13x1x3xf32>
  %0 = "tosa.reduce_min"(%arg0) {axis = 1 : i64} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
  return %0 : tensor<13x1x3xf32>
}

func @test_reduce_prod(%arg0: tensor<13x21x3xf32>) -> tensor<1x21x3xf32> {
  // CHECK: %0 = emitc.call "tosa::reduce_prod"(%arg0) {args = [0 : index, 0], template_args = [tensor<21x3xf32>, tensor<13x21x3xf32>]} : (tensor<13x21x3xf32>) -> tensor<21x3xf32>
  // CHECK: %1 = emitc.call "tosa::reshape"(%0) {template_args = [tensor<1x21x3xf32>]} : (tensor<21x3xf32>) -> tensor<1x21x3xf32>
  %0 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  return %0 : tensor<1x21x3xf32>
}

func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<13x1x3xf32> {
  // CHECK: %0 = emitc.call "tosa::reduce_sum"(%arg0) {args = [0 : index, 1], template_args = [tensor<13x3xf32>, tensor<13x21x3xf32>]} : (tensor<13x21x3xf32>) -> tensor<13x3xf32>
  // CHECK: %1 = emitc.call "tosa::reshape"(%0) {template_args = [tensor<13x1x3xf32>]} : (tensor<13x3xf32>) -> tensor<13x1x3xf32>
  %0 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
  return %0 : tensor<13x1x3xf32>
}

func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  // CHECK: %0 = emitc.call "tosa::reshape"(%arg0) {template_args = [tensor<1x819xf32>]} : (tensor<13x21x3xf32>) -> tensor<1x819xf32>
  %0 = "tosa.reshape"(%arg0) {new_shape = [1, 819]} : (tensor<13x21x3xf32>) -> tensor<1x819xf32>
  return %0 : tensor<1x819xf32>
}
