// RUN: emitc-opt -preprocess-mhlo-for-emitc %s | emitc-opt -convert-mhlo-to-emitc | emitc-translate --mlir-to-cpp | FileCheck %s

func @log_plus_one(%input: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo::add
  // CHECK: mhlo::log
  %0 = "mhlo.log_plus_one"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %0: tensor<4xf32>
}

func @exponential_minus_one(%input: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo::exp
  // CHECK: mhlo::sub
  %0 = "mhlo.exponential_minus_one"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %0: tensor<4xf32>
}
