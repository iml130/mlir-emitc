// RUN: emitc-opt -convert-mhlo-to-emitc %s | FileCheck %s

func @mhlo_addi(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: emitc.call "mhlo::add"
  %0 = mhlo.add %arg0, %arg0 : tensor<i64>
  return %0 : tensor<i64>
}

func @mhlo_addf(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK: emitc.call "mhlo::add"
  %0 = mhlo.add %arg0, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

func @mhlo_sub(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call "mhlo::sub"
  %0 = "mhlo.subtract"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @mhlo_multiply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call "mhlo::mul"
  %0 = "mhlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @mhlo_divide(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: emitc.call "mhlo::div"
  %0 = "mhlo.divide"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @mhlo_cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: emitc.call "mhlo::cos"
  %0 = "mhlo.cosine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @mhlo_concaternate(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>) -> tensor<3xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<1xf32>, tensor<2xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

func @mhlo_convert(%arg0: tensor<ui32>) -> tensor<ui64> {
  %0 = "mhlo.convert"(%arg0) : (tensor<ui32>) -> tensor<ui64>
  return %0 : tensor<ui64>
}
