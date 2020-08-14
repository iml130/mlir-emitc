func @mhlo_addi(%arg0: tensor<i64>) -> tensor<i64> {
  %0 = mhlo.add %arg0, %arg0 : tensor<i64>
  return %0 : tensor<i64>
}

func @mhlo_addf(%arg0: tensor<f64>) -> tensor<f64> {
  %0 = mhlo.add %arg0, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}
