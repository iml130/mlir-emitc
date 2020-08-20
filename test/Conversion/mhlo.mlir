// RUN: emitc-opt -convert-mhlo-to-emitc %s | FileCheck %s

func @float_abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call "mhlo::abs"
  %0 = "mhlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @mhlo_convert(%arg0: tensor<ui32>) -> tensor<ui64> {
  // CHECK: emitc.call "mhlo::convert"
  %0 = "mhlo.convert"(%arg0) : (tensor<ui32>) -> tensor<ui64>
  return %0 : tensor<ui64>
}

func @mhlo_cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call "mhlo::cos"
  %0 = "mhlo.cosine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @mhlo_exponential(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call "mhlo::exponential"
  %0 = "mhlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @mhlo_is_finite(%arg0: tensor<4xf32>) -> tensor<4xi1> {
  // CHECK: emitc.call "mhlo::isfinite"
  %0 = "mhlo.is_finite"(%arg0) : (tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

func @mhlo_log(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call "mhlo::log"
  %0 = "mhlo.log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @mhlo_negate(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call "mhlo::negate"
  %0 = "mhlo.negate"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @mhlo_add_i64(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: emitc.call "mhlo::add"
  %0 = mhlo.add %arg0, %arg0 : tensor<i64>
  return %0 : tensor<i64>
}

func @mhlo_add_f64(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK: emitc.call "mhlo::add"
  %0 = mhlo.add %arg0, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

func @mhlo_divide(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call "mhlo::div"
  %0 = "mhlo.divide"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @mhlo_max(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: emitc.call "mhlo::max"
  %0 = "mhlo.maximum"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func @mhlo_min(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: emitc.call "mhlo::min"
  %0 = "mhlo.minimum"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func @mhlo_multiply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call "mhlo::mul"
  %0 = "mhlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @mhlo_power(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call "mhlo::pow"
  %0 = "mhlo.power"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @mhlo_shift_left(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call "mhlo::shift_left"
  %0 = "mhlo.shift_left"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @mhlo_shift_right_logical(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call "mhlo::shift_right_logical"
  %0 = "mhlo.shift_right_logical"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @mhlo_sub(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call "mhlo::sub"
  %0 = "mhlo.subtract"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @mhlo_or(%arg0: tensor<ui64>, %arg1: tensor<ui64>) -> tensor<ui64> {
  // CHECK: emitc.call "mhlo::or"
  %0 = "mhlo.or"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  return %0 : tensor<ui64>
}

func @mhlo_xor(%arg0: tensor<ui64>, %arg1: tensor<ui64>) -> tensor<ui64> {
  // CHECK: emitc.call "mhlo::xor"
  %0 = "mhlo.xor"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  return %0 : tensor<ui64>
}

func @mhlo_tuple(%arg0: tensor<i32>, %arg1: tensor<ui64>) -> () {
  // CHECK: emitc.call "std::make_tuple"() {args = []}
  %0 = "mhlo.tuple"() : () -> tuple<>
  // CHECK: emitc.call "std::make_tuple"(%arg0) {args = [0 : index]}
  %1 = "mhlo.tuple"(%arg0) : (tensor<i32>) -> tuple<tensor<i32>>
  // CHECK: emitc.call "std::make_tuple"(%arg0, %arg1) {args = [0 : index, 1 : index]}
  %2 = "mhlo.tuple"(%arg0, %arg1) : (tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tensor<ui64>>
  // CHECK: emitc.call "std::make_tuple"(%arg0, %arg1, %arg0, %arg1) {args = [0 : index, 1 : index, 2 : index, 3 : index]}
  %3 = "mhlo.tuple"(%arg0, %arg1, %arg0, %arg1) : (tensor<i32>, tensor<ui64>, tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tensor<ui64>, tensor<i32>, tensor<ui64>>
  return
}

func @mhlo_tuple_recursive(%arg0: tensor<i32>, %arg1: tensor<ui64>) -> () {
  %1 = "mhlo.tuple"(%arg0, %arg1) : (tensor<i32>, tensor<ui64>) -> tuple<tensor<i32>, tensor<ui64>>
  %2 = "mhlo.tuple"(%arg0, %1) : (tensor<i32>, tuple<tensor<i32>, tensor<ui64>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tensor<ui64>>>
  return
}

func @mhlo_concaternate(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>) -> tensor<3xf32> {
  // CHECK: emitc.call "mhlo::concatenate"
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
