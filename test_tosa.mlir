module  {
  func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = emitc.call "tosa::abs"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = emitc.call "tosa::exp"(%arg0) {template_args = []} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = emitc.call "tosa::add"(%arg0, %arg1) {template_args = []} : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
  func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
    %0 = emitc.call "tosa::conv2D"(%arg0, %arg1, %arg2) {args = [0 : index, 1 : index, 2 : index, [0, 0, 0, 0], [1, 1], [1, 1]], template_args = [tensor<1x4x4x8xf32>, tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>]} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>
    return %0 : tensor<1x4x4x8xf32>
  }
}
