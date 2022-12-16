// RUN: emitc-opt -convert-tosa-to-emitc %s | FileCheck %s
// RUN: emitc-opt -insert-emitc-tosa-include -convert-tosa-to-emitc %s | FileCheck %s --check-prefixes=CHECK,CHECK-INCLUDE
// RUN: emitc-opt -tosa-to-emitc-pipeline %s | FileCheck %s --check-prefixes=CHECK,CHECK-INCLUDE

// CHECK-INCLUDE: emitc.include "emitc/tosa.h"

// Data node ops

func.func @test_const(%arg0 : index) -> tensor<4xi32> {
    // CHECK: "emitc.constant"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %0 = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    return %0 : tensor<4xi32>
}


// Unary elementwise ops

func.func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::tosa::abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// CHECK-LABEL: cast
func.func @test_cast(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::cast"(%arg0) {template_args = [tensor<13x21x3xf32>]} : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
  %0 = "tosa.cast"(%arg0) : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::tosa::ceil"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.ceil"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_clamp0(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::clamp"(%arg0) {args = [0 : index, 0.000000e+00 : f32, 1.000000e+00 : f32]} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.clamp"(%arg0) {min_fp = 0.0 : f32, max_fp = 1.0 : f32, min_int = -2 : i64, max_int = 2 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_clamp1(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: %0 = emitc.call "emitc::tosa::clamp"(%arg0) {args = [0 : index, -2 : i32, 2 : i32]} : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.clamp"(%arg0) {min_fp = 0.0 : f32, max_fp = 1.0 : f32, min_int = -2 : i64, max_int = 2 : i64} : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

func.func @test_clz(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::tosa::clz"(%arg0) : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.clz"(%arg0) : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

func.func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::tosa::exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::tosa::floor"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.floor"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::tosa::log"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.log"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::negate"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.negate"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_reciprocal(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::reciprocal"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.reciprocal"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_rescale(%arg0: tensor<13x21x3xui8>) -> tensor<13x21x3xi8> {
  // CHECK: %0 = emitc.call "emitc::tosa::rescale"(%arg0) {args = [0 : index, 127 : i32, -1 : i32, dense<1073741824> : tensor<1xi64>, dense<30> : tensor<1xi64>, true, false, false], template_args = [tensor<13x21x3xi8>, 1 : i32]} : (tensor<13x21x3xui8>) -> tensor<13x21x3xi8>
  %0 = "tosa.rescale"(%arg0) {double_round = false, input_zp = 127 : i32, multiplier = [1073741824 : i32], output_zp = -1 : i32, per_channel = false, scale32 = true, shift = [30 : i32]} : (tensor<13x21x3xui8>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

func.func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::sqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  // CHECK: %1 = emitc.call "emitc::tosa::reciprocal"(%0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_tanh(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::tanh"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.tanh"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// Binary elementwise ops

func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  // CHECK: emitc.call "emitc::tosa::add"(%0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// ArithmeticRightShiftOp: no broadcast
func.func @test_arithmetic_right_shift1(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::tosa::arithmetic_right_shift"(%arg0, %arg1) {args = [0 : index, 1 : index, false]} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.arithmetic_right_shift"(%arg0, %arg1) { round = false } : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// ArithmeticRightShiftOp: First operand needs to be broadcasted
func.func @test_arithmetic_right_shift2(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xi32>]} : (tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  // CHECK: emitc.call "emitc::tosa::arithmetic_right_shift"(%0, %arg1) {args = [0 : index, 1 : index, true]} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.arithmetic_right_shift"(%arg0, %arg1) { round = true } : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

func.func @test_equal(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi1> {
  // CHECK: emitc.call "emitc::tosa::equal"(%arg0, %arg1) {template_args = [tensor<13x21x3xi1>]} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi1>
  %0 = "tosa.equal"(%arg0, %arg1) : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

func.func @test_logical_left_shift(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xi32>]} : (tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  // CHECK: emitc.call "emitc::tosa::logical_left_shift"(%0, %arg1) : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.logical_left_shift"(%arg0, %arg1) : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// MulOp: no broadcast
func.func @test_mul10(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::tosa::mul"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 0 : i32 } : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// MulOp: First operand needs to be broadcasted
func.func @test_mul1(%arg0: tensor<13x1x3xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xi32>]} : (tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  // CHECK: emitc.call "emitc::tosa::mul"(%0, %arg1) {args = [0 : index, 1 : index, 1 : i32]} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x1x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// MulOp: Second operand needs to be broadcasted
func.func @test_mul2(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xi32>]} : (tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  // CHECK: emitc.call "emitc::tosa::mul"(%arg0, %0) {args = [0 : index, 1 : index, 1 : i32]} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// MulOp: Second operand needs to be broadcasted + expanded to two dimensions
func.func @test_mul3(%arg0: tensor<21x3xi32>, %arg1: tensor<3xi32>) -> tensor<21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [0 : index, dense<1> : tensor<1xi64>], template_args = [tensor<21x3xi32>]} : (tensor<3xi32>) -> tensor<21x3xi32>
  // CHECK: emitc.call "emitc::tosa::mul"(%arg0, %0) {args = [0 : index, 1 : index, 3 : i32]} : (tensor<21x3xi32>, tensor<21x3xi32>) -> tensor<21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 3 : i32 } : (tensor<21x3xi32>, tensor<3xi32>) -> tensor<21x3xi32>
  return %0 : tensor<21x3xi32>
}

// MulOp: Second operand needs to be broadcasted + expanded to three dimensions
func.func @test_mul4(%arg0: tensor<13x21x3xi32>, %arg1: tensor<3xi32>) -> tensor<13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [0 : index, dense<2> : tensor<1xi64>], template_args = [tensor<13x21x3xi32>]} : (tensor<3xi32>) -> tensor<13x21x3xi32>
  // CHECK: emitc.call "emitc::tosa::mul"(%arg0, %0) {args = [0 : index, 1 : index, 1 : i32]} : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x21x3xi32>, tensor<3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// MulOp: Second two dimensional operand needs to be broadcasted + expanded to four dimensions
func.func @test_mul5(%arg0: tensor<2x13x21x3xi32>, %arg1: tensor<21x3xi32>) -> tensor<2x13x21x3xi32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [0 : index, dense<[2, 3]> : tensor<2xi64>], template_args = [tensor<2x13x21x3xi32>]} : (tensor<21x3xi32>) -> tensor<2x13x21x3xi32>
  // CHECK: emitc.call "emitc::tosa::mul"(%arg0, %0) {args = [0 : index, 1 : index, 5 : i32]} : (tensor<2x13x21x3xi32>, tensor<2x13x21x3xi32>) -> tensor<2x13x21x3xi32>
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 5 : i32 } : (tensor<2x13x21x3xi32>, tensor<21x3xi32>) -> tensor<2x13x21x3xi32>
  return %0 : tensor<2x13x21x3xi32>
}

func.func @test_maximum(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  // CHECK: %1 = emitc.call "emitc::tosa::maximum"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.maximum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_minimum(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<1x21x3xf32>) -> tensor<13x21x3xf32>
  // CHECK: %1 = emitc.call "emitc::tosa::minimum"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.minimum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_sub(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  // CHECK: emitc.call "emitc::tosa::sub"(%0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<13x21x3xf32>]} : (tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  // CHECK: %1 = emitc.call "emitc::tosa::pow"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = "tosa.pow"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func.func @test_table(%arg0: tensor<64xi16>, %arg1: tensor<513xi16>) -> tensor<64xi32> {
  // CHECK: emitc.call "emitc::tosa::table"(%arg0, %arg1) : (tensor<64xi16>, tensor<513xi16>) -> tensor<64xi32>
  %0 = "tosa.table"(%arg0, %arg1) : (tensor<64xi16>, tensor<513xi16>) -> tensor<64xi32>
  return %0 : tensor<64xi32>
}

// Ternary ops

func.func @test_select(%arg0: tensor<12x6x3xi1>, %arg1: tensor<12x6x3xi32>, %arg2: tensor<12x6x3xi32>) -> tensor<12x6x3xi32> {
  // CHECK: emitc.call "emitc::tosa::select"(%arg0, %arg1, %arg2) : (tensor<12x6x3xi1>, tensor<12x6x3xi32>, tensor<12x6x3xi32>) -> tensor<12x6x3xi32>
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<12x6x3xi1>, tensor<12x6x3xi32>, tensor<12x6x3xi32>) -> tensor<12x6x3xi32>
  return %0 : tensor<12x6x3xi32>
}

func.func @test_select_broadcast_condition(%arg0: tensor<12x6x1xi1>, %arg1: tensor<12x6x3xi32>, %arg2: tensor<12x6x3xi32>) -> tensor<12x6x3xi32> {
  // CHECK: %0 = emitc.call "emitc::broadcast_in_dim"(%arg0) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<12x6x3xi1>]} : (tensor<12x6x1xi1>) -> tensor<12x6x3xi1>
  // CHECK: %1 = emitc.call "emitc::tosa::select"(%0, %arg1, %arg2) : (tensor<12x6x3xi1>, tensor<12x6x3xi32>, tensor<12x6x3xi32>) -> tensor<12x6x3xi32>
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<12x6x1xi1>, tensor<12x6x3xi32>, tensor<12x6x3xi32>) -> tensor<12x6x3xi32>
  return %0 : tensor<12x6x3xi32>
}

func.func @test_select_broadcast_input(%arg0: tensor<12x6x3xi1>, %arg1: tensor<12x1x3xi32>, %arg2: tensor<12x6x3xi32>) -> tensor<12x6x3xi32> {
  // CHECK: %0 = emitc.call "emitc::broadcast_in_dim"(%arg1) {args = [0 : index, dense<[0, 1, 2]> : tensor<3xi64>], template_args = [tensor<12x6x3xi32>]} : (tensor<12x1x3xi32>) -> tensor<12x6x3xi32>
  // CHECK: %1 = emitc.call "emitc::tosa::select"(%arg0, %0, %arg2) : (tensor<12x6x3xi1>, tensor<12x6x3xi32>, tensor<12x6x3xi32>) -> tensor<12x6x3xi32>
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<12x6x3xi1>, tensor<12x1x3xi32>, tensor<12x6x3xi32>) -> tensor<12x6x3xi32>
  return %0 : tensor<12x6x3xi32>
} 

// Other ops

func.func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<52x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::concat"(%arg0, %arg1, %arg2, %arg3) {template_args = [0, tensor<52x21x3xf32>]} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<52x21x3xf32>
  %0 = "tosa.concat"(%arg0, %arg1, %arg2, %arg3) {axis = 0} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<52x21x3xf32>
  return %0 : tensor<52x21x3xf32>
}

func.func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
    // CHECK: %0 = emitc.call "emitc::tosa::conv2d"(%arg0, %arg1) {args = [0 : index, 1 : index, dense<0> : tensor<4xi64>, dense<1> : tensor<2xi64>, dense<1> : tensor<2xi64>], template_args = [tensor<1x4x4x8xf32>]} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32>
    // CHECK: %1 = emitc.call "emitc::broadcast_in_dim"(%arg2) {args = [0 : index, dense<3> : tensor<1xi64>], template_args = [tensor<1x4x4x8xf32>]} : (tensor<8xf32>) -> tensor<1x4x4x8xf32>
    // CHECK: %2 = emitc.call "emitc::tosa::add"(%0, %1) : (tensor<1x4x4x8xf32>, tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>
    return %0 : tensor<1x4x4x8xf32>
}

func.func @test_depthwise_conv2d(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<2x2x2x2xf32>, %arg2: tensor<4xf32>) -> tensor<1x3x4x4xf32> {
    // CHECK: %0 = emitc.call "emitc::tosa::depthwise_conv2d"(%arg0, %arg1) {args = [0 : index, 1 : index, dense<0> : tensor<4xi64>, dense<1> : tensor<2xi64>, dense<1> : tensor<2xi64>], template_args = [tensor<1x3x4x4xf32>]} : (tensor<1x4x5x2xf32>, tensor<2x2x2x2xf32>) -> tensor<1x3x4x4xf32>
    // CHECK: %1 = emitc.call "emitc::broadcast_in_dim"(%arg2) {args = [0 : index, dense<3> : tensor<1xi64>], template_args = [tensor<1x3x4x4xf32>]} : (tensor<4xf32>) -> tensor<1x3x4x4xf32>
    // CHECK: %2 = emitc.call "emitc::tosa::add"(%0, %1) : (tensor<1x3x4x4xf32>, tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x4x5x2xf32>, tensor<2x2x2x2xf32>, tensor<4xf32>) -> tensor<1x3x4x4xf32>
    return %0 : tensor<1x3x4x4xf32>
}

func.func @test_fully_connected(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>, %arg2: tensor<28xf32>) -> tensor<14x28xf32> {
  // CHECK: emitc.call "emitc::tosa::fully_connected"(%arg0, %arg1, %arg2) {template_args = [tensor<14x28xf32>]} : (tensor<14x19xf32>, tensor<19x28xf32>, tensor<28xf32>) -> tensor<14x28xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<14x19xf32>, tensor<19x28xf32>, tensor<28xf32>) -> tensor<14x28xf32>
  return %0 : tensor<14x28xf32>
}

func.func @test_gather(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x6xi32>) -> tensor<3x6x5xf32> {
  // CHECK: emitc.call "emitc::tosa::gather"(%arg0, %arg1) {template_args = [tensor<3x6x5xf32>]} : (tensor<3x4x5xf32>, tensor<3x6xi32>) -> tensor<3x6x5xf32>
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x6xi32>) -> tensor<3x6x5xf32>
  return %0 : tensor<3x6x5xf32>
}

func.func @test_matmul(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>) -> tensor<1x14x28xf32> {
  // CHECK: emitc.call "emitc::tosa::matmul"(%arg0, %arg1) : (tensor<1x14x19xf32>, tensor<1x19x28xf32>) -> tensor<1x14x28xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x14x19xf32>, tensor<1x19x28xf32>) -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

func.func @test_argmax(%arg0: tensor<13x21x3xi32>) -> tensor<21x3xi32> {
  // CHECK: %0 = emitc.call "emitc::tosa::argmax"(%arg0) {args = [0 : index, 0], template_args = [tensor<21x3xi32>, tensor<13x21x3xi32>]} : (tensor<13x21x3xi32>) -> tensor<21x3xi32>
  %0 = "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xi32>) -> tensor<21x3xi32>
  return %0 : tensor<21x3xi32>
}

// Reduce ops

func.func @test_reduce_all(%arg0: tensor<13x21x3xi1>) -> tensor<1x21x3xi1> {
  // CHECK: %0 = emitc.call "emitc::tosa::reduce_all"(%arg0) {args = [0 : index, 0], template_args = [tensor<21x3xi1>, tensor<13x21x3xi1>]} : (tensor<13x21x3xi1>) -> tensor<21x3xi1>
  // CHECK: %1 = emitc.call "emitc::tosa::reshape"(%0) {template_args = [tensor<1x21x3xi1>]} : (tensor<21x3xi1>) -> tensor<1x21x3xi1>
  %0 = "tosa.reduce_all"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

func.func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<13x1x3xi1> {
  // CHECK: %0 = emitc.call "emitc::tosa::reduce_any"(%arg0) {args = [0 : index, 1], template_args = [tensor<13x3xi1>, tensor<13x21x3xi1>]} : (tensor<13x21x3xi1>) -> tensor<13x3xi1>
  // %1 = emitc.call "emitc::tosa::reshape"(%0) {template_args = [tensor<13x1x3xi1>]} : (tensor<13x3xi1>) -> tensor<13x1x3xi1>
  %0 = "tosa.reduce_any"(%arg0) {axis = 1 : i64} : (tensor<13x21x3xi1>) -> tensor<13x1x3xi1>
  return %0 : tensor<13x1x3xi1>
}

func.func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<1x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::reduce_max"(%arg0) {args = [0 : index, 0], template_args = [tensor<21x3xf32>, tensor<13x21x3xf32>]} : (tensor<13x21x3xf32>) -> tensor<21x3xf32>
  // CHECK: %1 = emitc.call "emitc::tosa::reshape"(%0) {template_args = [tensor<1x21x3xf32>]} : (tensor<21x3xf32>) -> tensor<1x21x3xf32>
  %0 = "tosa.reduce_max"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  return %0 : tensor<1x21x3xf32>
}

func.func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<13x1x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::reduce_min"(%arg0) {args = [0 : index, 1], template_args = [tensor<13x3xf32>, tensor<13x21x3xf32>]} : (tensor<13x21x3xf32>) -> tensor<13x3xf32>
  // CHECK: %1 = emitc.call "emitc::tosa::reshape"(%0) {template_args = [tensor<13x1x3xf32>]} : (tensor<13x3xf32>) -> tensor<13x1x3xf32>
  %0 = "tosa.reduce_min"(%arg0) {axis = 1 : i64} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
  return %0 : tensor<13x1x3xf32>
}

func.func @test_reduce_prod(%arg0: tensor<13x21x3xf32>) -> tensor<1x21x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::reduce_prod"(%arg0) {args = [0 : index, 0], template_args = [tensor<21x3xf32>, tensor<13x21x3xf32>]} : (tensor<13x21x3xf32>) -> tensor<21x3xf32>
  // CHECK: %1 = emitc.call "emitc::tosa::reshape"(%0) {template_args = [tensor<1x21x3xf32>]} : (tensor<21x3xf32>) -> tensor<1x21x3xf32>
  %0 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  return %0 : tensor<1x21x3xf32>
}

func.func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<13x1x3xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::reduce_sum"(%arg0) {args = [0 : index, 1], template_args = [tensor<13x3xf32>, tensor<13x21x3xf32>]} : (tensor<13x21x3xf32>) -> tensor<13x3xf32>
  // CHECK: %1 = emitc.call "emitc::tosa::reshape"(%0) {template_args = [tensor<13x1x3xf32>]} : (tensor<13x3xf32>) -> tensor<13x1x3xf32>
  %0 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
  return %0 : tensor<13x1x3xf32>
}

func.func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<4x11x1xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::slice"(%arg0) {args = [0 : index, dense<[6, 8, 0]> : tensor<3xi64>, dense<[4, 11, 1]> : tensor<3xi64>], template_args = [tensor<4x11x1xf32>]} : (tensor<13x21x3xf32>) -> tensor<4x11x1xf32>
  %0 = "tosa.slice"(%arg0) {start = [6, 8, 0], size = [4, 11, 1]} : (tensor<13x21x3xf32>) -> tensor<4x11x1xf32>
  return %0 : tensor<4x11x1xf32>
}

func.func @test_pad(%arg0: tensor<2x3xf32>, %arg1: tensor<2x2xi32>) -> tensor<3x6xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::pad"(%arg0, %arg1) {template_args = [tensor<3x6xf32>]} : (tensor<2x3xf32>, tensor<2x2xi32>) -> tensor<3x6xf32>
  %0 = "tosa.pad"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x2xi32>) -> tensor<3x6xf32>
  return %0 : tensor<3x6xf32>
}

func.func @test_pad_explicit_value(%arg0: tensor<2x3xf32>, %arg1: tensor<2x2xi32>) -> tensor<3x6xf32> {
  // CHECK: %0 = "emitc.constant"() {value = dense<3.140000e+00> : tensor<f32>} : () -> tensor<f32>
  %0 = "tosa.const"() {value = dense<3.14> : tensor<f32>} : () -> tensor<f32>
  // CHECK-NEXT: %1 = emitc.call "emitc::tosa::pad"(%arg0, %arg1, %0) {template_args = [tensor<3x6xf32>]} : (tensor<2x3xf32>, tensor<2x2xi32>, tensor<f32>) -> tensor<3x6xf32>
  %1 = "tosa.pad"(%arg0, %arg1, %0) : (tensor<2x3xf32>, tensor<2x2xi32>, tensor<f32>) -> tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
}

func.func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  // CHECK: %0 = emitc.call "emitc::tosa::reshape"(%arg0) {template_args = [tensor<1x819xf32>]} : (tensor<13x21x3xf32>) -> tensor<1x819xf32>
  %0 = "tosa.reshape"(%arg0) {new_shape = [1, 819]} : (tensor<13x21x3xf32>) -> tensor<1x819xf32>
  return %0 : tensor<1x819xf32>
}

func.func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
  // CHECK: %0 = "emitc.constant"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %0 = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT: %1 = emitc.call "emitc::tosa::transpose"(%arg0, %0) {template_args = [tensor<3x13x21xf32>]} : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  return %1 : tensor<3x13x21xf32>
}
