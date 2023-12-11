// RUN: emitc-opt -convert-stablehlo-region-ops-to-emitc -convert-stablehlo-to-emitc %s | FileCheck %s
// RUN: emitc-opt --insert-emitc-stablehlo-include -convert-stablehlo-region-ops-to-emitc -convert-stablehlo-to-emitc %s | FileCheck %s  --check-prefixes=CHECK,CHECK-INCLUDE
// RUN: emitc-opt -stablehlo-to-emitc-pipeline %s | FileCheck %s --check-prefixes=CHECK,CHECK-INCLUDE

// CHECK-INCLUDE: emitc.include "emitc/stablehlo.h"

// Nullary ops

func.func @stablehlo_constant(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: "emitc.constant"() <{value = dense<1> : tensor<2xi32>}> : () -> tensor<2xi32>
  %0 = "stablehlo.constant"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  return %0 : tensor<2xi32>
}


// Unary elementwise ops


func.func @float_abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_ceil(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_convert(%arg0: tensor<ui32>) -> tensor<ui64> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::convert"(%arg0) {template_args = [tensor<ui64>]} : (tensor<ui32>) -> tensor<ui64>
  %0 = "stablehlo.convert"(%arg0) : (tensor<ui32>) -> tensor<ui64>
  return %0 : tensor<ui64>
}

func.func @stablehlo_cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::cos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.cosine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_exponential(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_exponential_minus_one(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::exponential_minus_one"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_floor(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_is_finite(%arg0: tensor<4xf32>) -> tensor<4xi1> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::is_finite"(%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  %0 = "stablehlo.is_finite"(%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

func.func @stablehlo_log(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_log_plus_one(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::log_plus_one"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_negate(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::negate"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.negate"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_round(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::round"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.round_nearest_afz"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_sine(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::sin"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.sine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_sqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::sqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.sqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_tanh(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "stablehlo.tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}


// Binary elementwise ops

func.func @stablehlo_add_i64(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::add"(%arg0, %arg0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %0 = stablehlo.add %arg0, %arg0 : tensor<i64>
  return %0 : tensor<i64>
}

func.func @stablehlo_add_f64(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::add"(%arg0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  %0 = stablehlo.add %arg0, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

func.func @stablehlo_atan2(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::atan2"
  %0 = "stablehlo.atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @stablehlo_divide(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::div"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.divide"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @stablehlo_max(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::max"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %0 = "stablehlo.maximum"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @stablehlo_min(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::min"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %0 = "stablehlo.minimum"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @stablehlo_multiply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::mul"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @stablehlo_power(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::pow"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.power"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @stablehlo_shift_left(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::shift_left"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "stablehlo.shift_left"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

func.func @stablehlo_shift_right_logical(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::shift_right_logical"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "stablehlo.shift_right_logical"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

func.func @stablehlo_sub(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::sub"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.subtract"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}


// Binary logical elementwise ops

func.func @stablehlo_or(%arg0: tensor<ui64>, %arg1: tensor<ui64>) -> tensor<ui64> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::logical_or"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  %0 = "stablehlo.or"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  return %0 : tensor<ui64>
}

func.func @stablehlo_xor(%arg0: tensor<ui64>, %arg1: tensor<ui64>) -> tensor<ui64> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::logical_xor"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  %0 = "stablehlo.xor"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  return %0 : tensor<ui64>
}


// Tuple ops

func.func @stablehlo_tuple(%arg0: tensor<i32>, %arg1: tensor<ui64>) -> (tuple<tensor<i32>, tensor<ui64>, tensor<i32>, tensor<ui64>>) {
  // CHECK: emitc.call_opaque "std::make_tuple"() : () -> tuple<>
  %0 = "stablehlo.tuple"() : () -> tuple<>
  // CHECK: emitc.call_opaque "std::make_tuple"(%arg0) : (tensor<i32>) -> tuple<tensor<i32>>
  %1 = "stablehlo.tuple"(%arg0) : (tensor<i32>) -> tuple<tensor<i32>>
  // CHECK: emitc.call_opaque "std::make_tuple"(%arg0, %arg1) : (tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tensor<ui64>>
  %2 = "stablehlo.tuple"(%arg0, %arg1) : (tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tensor<ui64>>
  // CHECK: emitc.call_opaque "std::make_tuple"(%arg0, %arg1, %arg0, %arg1) : (tensor<i32>, tensor<ui64>, tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tensor<ui64>, tensor<i32>, tensor<ui64>>
  %3 = "stablehlo.tuple"(%arg0, %arg1, %arg0, %arg1) : (tensor<i32>, tensor<ui64>, tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tensor<ui64>, tensor<i32>, tensor<ui64>>
  return %3 : tuple<tensor<i32>, tensor<ui64>, tensor<i32>, tensor<ui64>>
}

func.func @stablehlo_tuple_nested(%arg0: tensor<i32>, %arg1: tensor<ui64>) -> tuple<tensor<i32>, tuple<tensor<i32>, tensor<ui64>>> {
  // CHECK: emitc.call_opaque "std::make_tuple"(%arg0, %arg1) : (tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tensor<ui64>>
  %0 = "stablehlo.tuple"(%arg0, %arg1) : (tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tensor<ui64>>
  // CHECK: emitc.call_opaque "std::make_tuple"(%arg0, %0) : (tensor<i32>, tuple<tensor<i32>, tensor<ui64>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tensor<ui64>>>
  %1 = "stablehlo.tuple"(%arg0, %0) : (tensor<i32>, tuple<tensor<i32>, tensor<ui64>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tensor<ui64>>>
  return %1 : tuple<tensor<i32>, tuple<tensor<i32>, tensor<ui64>>>
}

func.func @stablehlo_tuple_unpack(%arg0: tensor<i32>, %arg1: tensor<ui64>) -> (tuple<tensor<i32>, tensor<ui64>>, tensor<i32>) {
  %0 = call @stablehlo_tuple_nested(%arg0, %arg1) : (tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tuple<tensor<i32>, tensor<ui64>>>
  // CHECK: emitc.call_opaque "std::get"(%0) {template_args = [1 : i32]} : (tuple<tensor<i32>, tuple<tensor<i32>, tensor<ui64>>>) -> tuple<tensor<i32>, tensor<ui64>>
  %1 = "stablehlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tensor<ui64>>>) -> tuple<tensor<i32>, tensor<ui64>>
  // CHECK: emitc.call_opaque "std::get"(%1) {template_args = [0 : i32]} : (tuple<tensor<i32>, tensor<ui64>>) -> tensor<i32>
  %2 = "stablehlo.get_tuple_element"(%1) {index = 0 : i32} : (tuple<tensor<i32>, tensor<ui64>>) -> tensor<i32>
  return %1, %2 : tuple<tensor<i32>, tensor<ui64>>, tensor<i32>
}

func.func @stablehlo_compare(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi1> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::compare"(%arg0, %arg1) {template_args = [tensor<4xi32>, #emitc.opaque<"std::less">]} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK: emitc.call_opaque "emitc::stablehlo::compare"(%arg0, %arg1) {template_args = [tensor<4xi32>, #emitc.opaque<"std::less_equal">]} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %1 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction LE>} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK: emitc.call_opaque "emitc::stablehlo::compare"(%arg0, %arg1) {template_args = [tensor<4xi32>, #emitc.opaque<"std::greater">]} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %2 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK: emitc.call_opaque "emitc::stablehlo::compare"(%arg0, %arg1) {template_args = [tensor<4xi32>, #emitc.opaque<"std::greater_equal">]} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %3 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK: emitc.call_opaque "emitc::stablehlo::compare"(%arg0, %arg1) {template_args = [tensor<4xi32>, #emitc.opaque<"std::equal_to">]} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %4 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK: emitc.call_opaque "emitc::stablehlo::compare"(%arg0, %arg1) {template_args = [tensor<4xi32>, #emitc.opaque<"std::not_equal_to">]} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %5 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>

  return %0 : tensor<4xi1>
}


// Slice ops

func.func @stablehlo_slice(%arg0: tensor<12xi32>, %arg1: tensor<8x7xi32>) -> tensor<4x3xi32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::slice"(%arg0) {args = [0 : index, dense<0> : tensor<1xi64>, dense<1> : tensor<1xi64>, dense<1> : tensor<1xi64>], template_args = [tensor<1xi32>]} : (tensor<12xi32>) -> tensor<1xi32>
  %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<12xi32>) -> tensor<1xi32>
  // CHECK: emitc.call_opaque "emitc::stablehlo::slice"(%arg1) {args = [0 : index, dense<0> : tensor<2xi64>, dense<[4, 3]> : tensor<2xi64>, dense<1> : tensor<2xi64>], template_args = [tensor<4x3xi32>]} : (tensor<8x7xi32>) -> tensor<4x3xi32>
  %1 = "stablehlo.slice"(%arg1) {limit_indices = dense<[4, 3]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x7xi32>) -> tensor<4x3xi32>
  return %1 : tensor<4x3xi32>
}

func.func @stablehlo_dynamic_slice(%arg0: tensor<12xi32>, %arg1: tensor<8x7xi32>) -> () {
  %cst = "arith.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %cst_0 = "arith.constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
  // CHECK: emitc.call_opaque "emitc::stablehlo::dynamic_slice"(%arg0, %cst) {args = [0 : index, 1 : index, dense<4> : tensor<1xi64>], template_args = [tensor<4xi32>]} : (tensor<12xi32>, tensor<i64>) -> tensor<4xi32>
  %0 = "stablehlo.dynamic_slice"(%arg0, %cst) {slice_sizes = dense<4> : tensor<1xi64>} : (tensor<12xi32>, tensor<i64>) -> tensor<4xi32>
  // CHECK: emitc.call_opaque "emitc::stablehlo::dynamic_slice"(%arg1, %cst, %cst_0) {args = [0 : index, 1 : index, 2 : index, dense<[4, 2]> : tensor<2xi64>], template_args = [tensor<4x2xi32>]} : (tensor<8x7xi32>, tensor<i64>, tensor<i64>) -> tensor<4x2xi32>
  %1 = "stablehlo.dynamic_slice"(%arg1, %cst, %cst_0) {slice_sizes = dense<[4, 2]> : tensor<2xi64>} : (tensor<8x7xi32>, tensor<i64>, tensor<i64>) -> tensor<4x2xi32>
  return
}

func.func @stablehlo_dynamic_update_slice(%arg0: tensor<12xi32>, %arg1: tensor<8x7xi32>) -> () {
  %cst = "arith.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %cst_0 = "arith.constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
  %cst_1 = "arith.constant"() {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_2 = "arith.constant"() {value = dense<1> : tensor<2x4xi32>} : () -> tensor<2x4xi32>
  // CHECK: emitc.call_opaque "emitc::stablehlo::dynamic_update_slice"(%arg0, %cst_1, %cst) {template_args = [tensor<4xi32>]}
  %0 = "stablehlo.dynamic_update_slice"(%arg0, %cst_1, %cst) : (tensor<12xi32>, tensor<4xi32>, tensor<i64>) -> tensor<12xi32>
  // CHECK: emitc.call_opaque "emitc::stablehlo::dynamic_update_slice"(%arg1, %cst_2, %cst, %cst_0) {template_args = [tensor<2x4xi32>]}
  %1 = "stablehlo.dynamic_update_slice"(%arg1, %cst_2, %cst, %cst_0) : (tensor<8x7xi32>, tensor<2x4xi32>, tensor<i64>, tensor<i64>) -> tensor<8x7xi32>
  return
}


// Other ops

func.func @stablehlo_batch_norm_inference(%arg0: tensor<4x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<4x2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {args = [0 : index, 1 : index, 2 : index, 3 : index, 4 : index, 1.000000e-03 : f32, 1], template_args = [tensor<4x2xf32>, tensor<2xf32>]} : (tensor<4x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  %0 = "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 0.001 : f32, feature_index = 1 : i64} : (tensor<4x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

func.func @stablehlo_bitcast_convert(%arg0: tensor<ui32>) -> tensor<i32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::bitcast_convert"(%arg0) {template_args = [tensor<i32>]} : (tensor<ui32>) -> tensor<i32>
  %0 = "stablehlo.bitcast_convert"(%arg0) : (tensor<ui32>) -> tensor<i32>
  return %0 : tensor<i32>
}

func.func @stablehlo_broadcast_in_dim(%arg0: tensor<i32>) -> tensor<3xi32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::broadcast_in_dim"(%arg0) {args = [0 : index, dense<> : tensor<0xi64>], template_args = [tensor<3xi32>]} : (tensor<i32>) -> tensor<3xi32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

func.func @stablehlo_clamp(%arg0: tensor<2x1xf32>, %arg1: tensor<2x1xf32>, %arg2: tensor<2x1xf32>) -> tensor<2x1xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::clamp"(%arg0, %arg1, %arg2) {template_args = [tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>]} : (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %0 = "stablehlo.clamp"(%arg0, %arg1, %arg2) : (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

func.func @stablehlo_clamp_broadcast(%arg0: tensor<i32>, %arg1: tensor<4x2x1xi32>, %arg2: tensor<i32>) -> tensor<4x2x1xi32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::clamp"(%arg0, %arg1, %arg2) {template_args = [tensor<i32>, tensor<4x2x1xi32>, tensor<i32>]} : (tensor<i32>, tensor<4x2x1xi32>, tensor<i32>) -> tensor<4x2x1xi32>
  %0 = "stablehlo.clamp"(%arg0, %arg1, %arg2) : (tensor<i32>, tensor<4x2x1xi32>, tensor<i32>) -> tensor<4x2x1xi32>
  return %0 : tensor<4x2x1xi32>
}

func.func @stablehlo_concaternate(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>) -> tensor<3xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::concatenate"(%arg0, %arg1) {template_args = [0, tensor<3xf32>]} : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// Initially taken over from
// https://github.com/tensorflow/mlir-hlo/blob/31c1c3aa1ffa12b1fb2d9988ad8cc0b2de9cd581/tests/hlo-legalize-to-lhlo.mlir#L552-L580
// Updated to
// https://github.com/tensorflow/mlir-hlo/blob/2ee7bcc4b5d5a8afc7cbd414d10cdfeda97b7a54/tests/Dialect/stablehlo/hlo-legalize-to-lhlo.mlir#L475-L501
func.func @stablehlo_conv(%arg0: tensor<3x2x4x3xf32>, %arg1 : tensor<2x2x3x4xf32>) -> tensor<2x1x2x3xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: emitc.call_opaque "emitc::stablehlo::convolution"(%arg1, %arg0)
  %out = "stablehlo.convolution"(%arg1, %arg0) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<[1, 2]> : tensor<2xi64>,
    window_strides = dense<[2, 1]> : tensor<2xi64>
  } : (tensor<2x2x3x4xf32>, tensor<3x2x4x3xf32>) -> tensor<2x1x2x3xf32>
  return %out : tensor<2x1x2x3xf32>
}

func.func @stablehlo_dot(%arg0: tensor<512x512xf32>) -> tensor<512x512xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::dot"(%arg0, %arg0) {template_args = [tensor<512x512xf32>]} : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  %0 = "stablehlo.dot"(%arg0, %arg0) : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
  return %0 : tensor<512x512xf32>
}

func.func @stablehlo_pad(%arg0: tensor<2x3xf32>, %arg1: tensor<f32>) -> tensor<4x7xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::pad"(%arg0, %arg1) {args = [0 : index, 1 : index, dense<-1> : tensor<2xi64>, dense<1> : tensor<2xi64>, dense<2> : tensor<2xi64>], template_args = [tensor<4x7xf32>]} : (tensor<2x3xf32>, tensor<f32>) -> tensor<4x7xf32>
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<-1> : tensor<2xi64>,
    edge_padding_high = dense<1> : tensor<2xi64>,
    interior_padding = dense<2> : tensor<2xi64>
  } : (tensor<2x3xf32>, tensor<f32>) -> tensor<4x7xf32>
  return %0 : tensor<4x7xf32>
}

func.func @stablehlo_reduce(%arg0 : tensor<2x1000xf32>, %arg1 : tensor<f32>, %arg2 : tensor<2x1000xi32>, %arg3 : tensor<i32>) -> tensor<2xi32>{
  // CHECK: func @stablehlo_reduce_lambda_0(%arg0: tensor<f32>, %arg1: tensor<f32>)
  // CHECK: "emitc::stablehlo::add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: func @stablehlo_reduce_lambda_1(%arg0: tensor<i32>, %arg1: tensor<i32>)
  // CHECK: "emitc::stablehlo::max"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: func @stablehlo_reduce_lambda_2(%arg0: tensor<f32>, %arg1: tensor<i32>, %arg2: tensor<f32>, %arg3: tensor<i32>)
  // CHECK: "emitc::stablehlo::max"(%arg0, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: "emitc::stablehlo::min"(%arg1, %arg3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  
  // CHECK: emitc.call_opaque "emitc::stablehlo::reduce"(%arg0, %arg1) {args = [0 : index, 1 : index, dense<1> : tensor<1xi64>, @stablehlo_reduce_lambda_0], template_args = [tensor<2xf32>, 1]} : (tensor<2x1000xf32>, tensor<f32>) -> tensor<2xf32>
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %1 = stablehlo.add %arg4, %arg5 : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x1000xf32>, tensor<f32>) -> tensor<2xf32>
  
  // CHECK: emitc.call_opaque "emitc::stablehlo::reduce"(%arg2, %arg3) {args = [0 : index, 1 : index, dense<1> : tensor<1xi64>, @stablehlo_reduce_lambda_1], template_args = [tensor<2xi32>, 1]} : (tensor<2x1000xi32>, tensor<i32>) -> tensor<2xi32>
  %1 = "stablehlo.reduce"(%arg2, %arg3) ({
    ^bb0(%arg4: tensor<i32>, %arg5: tensor<i32>):
      %2 = stablehlo.maximum %arg4, %arg5 : tensor<i32>
      "stablehlo.return"(%2) : (tensor<i32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x1000xi32>, tensor<i32>) -> tensor<2xi32>
  
  // CHECK: emitc.call_opaque "emitc::stablehlo::reduce"(%arg0, %arg2, %arg1, %arg3) {args = [0 : index, 1 : index, 2 : index, 3 : index, dense<1> : tensor<1xi64>, @stablehlo_reduce_lambda_2], template_args = [tensor<2xf32>, tensor<2xi32>, 1]} : (tensor<2x1000xf32>, tensor<2x1000xi32>, tensor<f32>, tensor<i32>) -> (tensor<2xf32>, tensor<2xi32>)
  %2:2 = stablehlo.reduce(%arg0 init: %arg1), (%arg2 init: %arg3) across dimensions = [1] : (tensor<2x1000xf32>, tensor<2x1000xi32>, tensor<f32>, tensor<i32>) -> (tensor<2xf32>, tensor<2xi32>)
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>) (%arg6: tensor<i32>, %arg7: tensor<i32>)  {
      %2 = stablehlo.maximum %arg4, %arg5 : tensor<f32>
      %3 = "stablehlo.minimum"(%arg6, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
    }
  
  return %1 : tensor<2xi32>
}

func.func @stablehlo_reduce_window(%arg0 : tensor<2x114x114x64xf32>, %arg1 : tensor<f32>) -> tensor<2x56x56x64xf32> {
  // CHECK: func @stablehlo_reduce_window_lambda_0(%arg0: tensor<f32>, %arg1: tensor<f32>)
  // CHECK: "emitc::stablehlo::max"
  // CHECK: emitc.call_opaque "emitc::stablehlo::reduce_window"(%arg0, %arg1) {args = [0 : index, 1 : index, dense<[1, 3, 3, 1]> : tensor<4xi64>, dense<[1, 2, 2, 1]> : tensor<4xi64>, dense<1> : tensor<4xi64>, dense<1> : tensor<4xi64>, dense<0> : tensor<8xi64>, @stablehlo_reduce_window_lambda_0], template_args = [tensor<2x56x56x64xf32>]}
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
      %516 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      "stablehlo.return"(%516) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<2x114x114x64xf32>, tensor<f32>) -> tensor<2x56x56x64xf32>
  
  return %0 : tensor<2x56x56x64xf32>
}

func.func @stablehlo_reshape(%arg0: tensor<12xf32>) -> tensor<2x3x2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::reshape"(%arg0) {template_args = [tensor<2x3x2xf32>]} : (tensor<12xf32>) -> tensor<2x3x2xf32>
  %0 = "stablehlo.reshape"(%arg0) : (tensor<12xf32>) -> tensor<2x3x2xf32>
  return %0 : tensor<2x3x2xf32>
}

func.func @stablehlo_select(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xi1>) -> tensor<2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::select"(%arg2, %arg0, %arg1) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %1 = "stablehlo.select"(%arg2, %arg0, %arg1) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

func.func @select_scalar_pred(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

func.func @stablehlo_transpose(%arg0: tensor<2x3x4xf32>) -> tensor<4x3x2xf32> {
  // CHECK: emitc.call_opaque "emitc::stablehlo::transpose"(%arg0) {args = [0 : index, dense<[2, 1, 0]> : tensor<3xi64>], template_args = [tensor<4x3x2xf32>]} : (tensor<2x3x4xf32>) -> tensor<4x3x2xf32>
  %0 = "stablehlo.transpose"(%arg0) {permutation = dense<[2, 1, 0]> : tensor<3xi64>} : (tensor<2x3x4xf32>) -> tensor<4x3x2xf32>
  return %0 : tensor<4x3x2xf32>
}

// RNG ops

func.func @stablehlo_rng_uniform() -> () {
  %cst = "arith.constant"() {value = dense<-100> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "arith.constant"() {value = dense<100> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "arith.constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>

  // TODO: Fix or drop test #: emitc.call_opaque "emitc::stablehlo::rng_uniform"(%cst, %cst_0, %cst_1) {template_args = [tensor<2xi32>]} : (tensor<i32>, tensor<i32>, tensor<1xi64>) -> tensor<2xi32>
  //%0 = "stablehlo.rng"(%cst, %cst_0, %cst_1) {rng_distribution = #stablehlo.rng_distribution<UNIFORM>}: (tensor<i32>, tensor<i32>, tensor<1xi64>) -> tensor<2xi32>
  
  %cst_2 = "arith.constant"() {value = dense<-100.0> : tensor<f32>} : () -> tensor<f32>
  %cst_3 = "arith.constant"() {value = dense<100.0> : tensor<f32>} : () -> tensor<f32>
  %cst_4 = "arith.constant"() {value = dense<17> : tensor<1xi64>} : () -> tensor<1xi64>

  // TODO: Fix or drop test #: emitc.call_opaque "emitc::stablehlo::rng_uniform"(%cst_2, %cst_3, %cst_4) {template_args = [tensor<17xf32>]} : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<17xf32>
  //%1 = "stablehlo.rng"(%cst_2, %cst_3, %cst_4) {rng_distribution = #stablehlo.rng_distribution<UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<17xf32>
  return
}
