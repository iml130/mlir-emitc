// RUN: emitc-opt %s --convert-tosa-to-emitc | emitc-translate --mlir-to-cpp
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 812 : i32}, tf_saved_model.semantics}  {
  func.func @predict(%arg0: tensor<1x224x224x3xf32> {tf._user_specified_name = "args_0", tf_saved_model.index_path = [0]}) -> (tensor<1x1000xf32> {tf_saved_model.index_path = []}) attributes {tf._construction_context = "kEagerRuntime"} {
    %0 = "tosa.const"() {value = dense<0.0204081628> : tensor<f32>} : () -> tensor<f32>
    %1 = "tosa.const"() {value = dense<1.000000e-03> : tensor<1xf32>} : () -> tensor<1xf32>
    %2 = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %3 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1280xf32>} : () -> tensor<1280xf32>
    %4 = "tosa.const"() {value = dense<0.000000e+00> : tensor<320xf32>} : () -> tensor<320xf32>
    %5 = "tosa.const"() {value = dense<0.000000e+00> : tensor<960xf32>} : () -> tensor<960xf32>
    %6 = "tosa.const"() {value = dense<0.000000e+00> : tensor<160xf32>} : () -> tensor<160xf32>
    %7 = "tosa.const"() {value = dense<0.000000e+00> : tensor<576xf32>} : () -> tensor<576xf32>
    %8 = "tosa.const"() {value = dense<0.000000e+00> : tensor<96xf32>} : () -> tensor<96xf32>
    %9 = "tosa.const"() {value = dense<0.000000e+00> : tensor<384xf32>} : () -> tensor<384xf32>
    %10 = "tosa.const"() {value = dense<0.000000e+00> : tensor<64xf32>} : () -> tensor<64xf32>
    %11 = "tosa.const"() {value = dense<0.000000e+00> : tensor<192xf32>} : () -> tensor<192xf32>
    %12 = "tosa.const"() {value = dense<0.000000e+00> : tensor<32xf32>} : () -> tensor<32xf32>
    %13 = "tosa.const"() {value = dense<0.000000e+00> : tensor<144xf32>} : () -> tensor<144xf32>
    %14 = "tosa.const"() {value = dense<0.000000e+00> : tensor<24xf32>} : () -> tensor<24xf32>
    %15 = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %16 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
    %17 = "tosa.const"() {value = dense<5.000000e-01> : tensor<3x3x3x32xf32>} : () -> tensor<3x3x3x32xf32>
    %18 = "tosa.const"() {value = dense<5.000000e-01> : tensor<3x3x32x1xf32>} : () -> tensor<3x3x32x1xf32>
    %19 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x32x16xf32>} : () -> tensor<1x1x32x16xf32>
    %20 = "tosa.const"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16xf32>
    %21 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x16x96xf32>} : () -> tensor<1x1x16x96xf32>
    %22 = "tosa.const"() {value = dense<5.000000e-01> : tensor<3x3x96x1xf32>} : () -> tensor<3x3x96x1xf32>
    %23 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x96x24xf32>} : () -> tensor<1x1x96x24xf32>
    %24 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x144x24xf32>} : () -> tensor<1x1x144x24xf32>
    %25 = "tosa.const"() {value = dense<5.000000e-01> : tensor<24xf32>} : () -> tensor<24xf32>
    %26 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x24x144xf32>} : () -> tensor<1x1x24x144xf32>
    %27 = "tosa.const"() {value = dense<5.000000e-01> : tensor<3x3x144x1xf32>} : () -> tensor<3x3x144x1xf32>
    %28 = "tosa.const"() {value = dense<5.000000e-01> : tensor<144xf32>} : () -> tensor<144xf32>
    %29 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x144x32xf32>} : () -> tensor<1x1x144x32xf32>
    %30 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x192x32xf32>} : () -> tensor<1x1x192x32xf32>
    %31 = "tosa.const"() {value = dense<5.000000e-01> : tensor<32xf32>} : () -> tensor<32xf32>
    %32 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x32x192xf32>} : () -> tensor<1x1x32x192xf32>
    %33 = "tosa.const"() {value = dense<5.000000e-01> : tensor<3x3x192x1xf32>} : () -> tensor<3x3x192x1xf32>
    %34 = "tosa.const"() {value = dense<5.000000e-01> : tensor<192xf32>} : () -> tensor<192xf32>
    %35 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x192x64xf32>} : () -> tensor<1x1x192x64xf32>
    %36 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x384x64xf32>} : () -> tensor<1x1x384x64xf32>
    %37 = "tosa.const"() {value = dense<5.000000e-01> : tensor<64xf32>} : () -> tensor<64xf32>
    %38 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x64x384xf32>} : () -> tensor<1x1x64x384xf32>
    %39 = "tosa.const"() {value = dense<5.000000e-01> : tensor<3x3x384x1xf32>} : () -> tensor<3x3x384x1xf32>
    %40 = "tosa.const"() {value = dense<5.000000e-01> : tensor<384xf32>} : () -> tensor<384xf32>
    %41 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x384x96xf32>} : () -> tensor<1x1x384x96xf32>
    %42 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x576x96xf32>} : () -> tensor<1x1x576x96xf32>
    %43 = "tosa.const"() {value = dense<5.000000e-01> : tensor<96xf32>} : () -> tensor<96xf32>
    %44 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x96x576xf32>} : () -> tensor<1x1x96x576xf32>
    %45 = "tosa.const"() {value = dense<5.000000e-01> : tensor<3x3x576x1xf32>} : () -> tensor<3x3x576x1xf32>
    %46 = "tosa.const"() {value = dense<5.000000e-01> : tensor<576xf32>} : () -> tensor<576xf32>
    %47 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x576x160xf32>} : () -> tensor<1x1x576x160xf32>
    %48 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x960x160xf32>} : () -> tensor<1x1x960x160xf32>
    %49 = "tosa.const"() {value = dense<5.000000e-01> : tensor<160xf32>} : () -> tensor<160xf32>
    %50 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x160x960xf32>} : () -> tensor<1x1x160x960xf32>
    %51 = "tosa.const"() {value = dense<5.000000e-01> : tensor<3x3x960x1xf32>} : () -> tensor<3x3x960x1xf32>
    %52 = "tosa.const"() {value = dense<5.000000e-01> : tensor<960xf32>} : () -> tensor<960xf32>
    %53 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x960x320xf32>} : () -> tensor<1x1x960x320xf32>
    %54 = "tosa.const"() {value = dense<5.000000e-01> : tensor<320xf32>} : () -> tensor<320xf32>
    %55 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x320x1280xf32>} : () -> tensor<1x1x320x1280xf32>
    %56 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1280xf32>} : () -> tensor<1280xf32>
    %57 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1280x1000xf32>} : () -> tensor<1280x1000xf32>
    %58 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1000xf32>} : () -> tensor<1000xf32>
    %59 = "tosa.transpose"(%17, %2) : (tensor<3x3x3x32xf32>, tensor<4xi32>) -> tensor<32x3x3x3xf32>
    %60 = "tosa.conv2d"(%arg0, %59, %12) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %61 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %62 = "tosa.sub"(%60, %61) : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %63 = "tosa.add"(%31, %1) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %64 = "tosa.rsqrt"(%63) : (tensor<32xf32>) -> tensor<32xf32>
    %65 = "tosa.reshape"(%64) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %66 = "tosa.mul"(%62, %65) {shift = 0 : i32} : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %67 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %68 = "tosa.mul"(%66, %67) {shift = 0 : i32} : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %69 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %70 = "tosa.add"(%68, %69) : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %71 = "tosa.clamp"(%70) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %72 = "tosa.depthwise_conv2d"(%71, %18, %12) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x112x112x32xf32>, tensor<3x3x32x1xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %73 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %74 = "tosa.sub"(%72, %73) : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %75 = "tosa.add"(%31, %1) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %76 = "tosa.rsqrt"(%75) : (tensor<32xf32>) -> tensor<32xf32>
    %77 = "tosa.reshape"(%76) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %78 = "tosa.mul"(%74, %77) {shift = 0 : i32} : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %79 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %80 = "tosa.mul"(%78, %79) {shift = 0 : i32} : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %81 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %82 = "tosa.add"(%80, %81) : (tensor<1x112x112x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x112x112x32xf32>
    %83 = "tosa.clamp"(%82) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %84 = "tosa.transpose"(%19, %2) : (tensor<1x1x32x16xf32>, tensor<4xi32>) -> tensor<16x1x1x32xf32>
    %85 = "tosa.conv2d"(%83, %84, %15) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x112x112x32xf32>, tensor<16x1x1x32xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %86 = "tosa.reshape"(%20) {new_shape = array<i64: 1, 1, 1, 16>} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %87 = "tosa.sub"(%85, %86) : (tensor<1x112x112x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x112x112x16xf32>
    %88 = "tosa.add"(%20, %1) : (tensor<16xf32>, tensor<1xf32>) -> tensor<16xf32>
    %89 = "tosa.rsqrt"(%88) : (tensor<16xf32>) -> tensor<16xf32>
    %90 = "tosa.reshape"(%89) {new_shape = array<i64: 1, 1, 1, 16>} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %91 = "tosa.mul"(%87, %90) {shift = 0 : i32} : (tensor<1x112x112x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x112x112x16xf32>
    %92 = "tosa.reshape"(%20) {new_shape = array<i64: 1, 1, 1, 16>} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %93 = "tosa.mul"(%91, %92) {shift = 0 : i32} : (tensor<1x112x112x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x112x112x16xf32>
    %94 = "tosa.reshape"(%20) {new_shape = array<i64: 1, 1, 1, 16>} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %95 = "tosa.add"(%93, %94) : (tensor<1x112x112x16xf32>, tensor<1x1x1x16xf32>) -> tensor<1x112x112x16xf32>
    %96 = "tosa.transpose"(%21, %2) : (tensor<1x1x16x96xf32>, tensor<4xi32>) -> tensor<96x1x1x16xf32>
    %97 = "tosa.conv2d"(%95, %96, %8) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x112x112x16xf32>, tensor<96x1x1x16xf32>, tensor<96xf32>) -> tensor<1x112x112x96xf32>
    %98 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %99 = "tosa.sub"(%97, %98) : (tensor<1x112x112x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x112x112x96xf32>
    %100 = "tosa.add"(%43, %1) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %101 = "tosa.rsqrt"(%100) : (tensor<96xf32>) -> tensor<96xf32>
    %102 = "tosa.reshape"(%101) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %103 = "tosa.mul"(%99, %102) {shift = 0 : i32} : (tensor<1x112x112x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x112x112x96xf32>
    %104 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %105 = "tosa.mul"(%103, %104) {shift = 0 : i32} : (tensor<1x112x112x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x112x112x96xf32>
    %106 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %107 = "tosa.add"(%105, %106) : (tensor<1x112x112x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x112x112x96xf32>
    %108 = "tosa.clamp"(%107) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x112x112x96xf32>) -> tensor<1x112x112x96xf32>
    %109 = "tosa.pad"(%108, %16) : (tensor<1x112x112x96xf32>, tensor<4x2xi32>) -> tensor<1x113x113x96xf32>
    %110 = "tosa.depthwise_conv2d"(%109, %22, %8) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x113x113x96xf32>, tensor<3x3x96x1xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %111 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %112 = "tosa.sub"(%110, %111) : (tensor<1x56x56x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x56x56x96xf32>
    %113 = "tosa.add"(%43, %1) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %114 = "tosa.rsqrt"(%113) : (tensor<96xf32>) -> tensor<96xf32>
    %115 = "tosa.reshape"(%114) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %116 = "tosa.mul"(%112, %115) {shift = 0 : i32} : (tensor<1x56x56x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x56x56x96xf32>
    %117 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %118 = "tosa.mul"(%116, %117) {shift = 0 : i32} : (tensor<1x56x56x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x56x56x96xf32>
    %119 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %120 = "tosa.add"(%118, %119) : (tensor<1x56x56x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x56x56x96xf32>
    %121 = "tosa.clamp"(%120) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
    %122 = "tosa.transpose"(%23, %2) : (tensor<1x1x96x24xf32>, tensor<4xi32>) -> tensor<24x1x1x96xf32>
    %123 = "tosa.conv2d"(%121, %122, %14) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x56x56x96xf32>, tensor<24x1x1x96xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %124 = "tosa.reshape"(%25) {new_shape = array<i64: 1, 1, 1, 24>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %125 = "tosa.sub"(%123, %124) : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %126 = "tosa.add"(%25, %1) : (tensor<24xf32>, tensor<1xf32>) -> tensor<24xf32>
    %127 = "tosa.rsqrt"(%126) : (tensor<24xf32>) -> tensor<24xf32>
    %128 = "tosa.reshape"(%127) {new_shape = array<i64: 1, 1, 1, 24>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %129 = "tosa.mul"(%125, %128) {shift = 0 : i32} : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %130 = "tosa.reshape"(%25) {new_shape = array<i64: 1, 1, 1, 24>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %131 = "tosa.mul"(%129, %130) {shift = 0 : i32} : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %132 = "tosa.reshape"(%25) {new_shape = array<i64: 1, 1, 1, 24>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %133 = "tosa.add"(%131, %132) : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %134 = "tosa.transpose"(%26, %2) : (tensor<1x1x24x144xf32>, tensor<4xi32>) -> tensor<144x1x1x24xf32>
    %135 = "tosa.conv2d"(%133, %134, %13) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x56x56x24xf32>, tensor<144x1x1x24xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %136 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %137 = "tosa.sub"(%135, %136) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %138 = "tosa.add"(%28, %1) : (tensor<144xf32>, tensor<1xf32>) -> tensor<144xf32>
    %139 = "tosa.rsqrt"(%138) : (tensor<144xf32>) -> tensor<144xf32>
    %140 = "tosa.reshape"(%139) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %141 = "tosa.mul"(%137, %140) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %142 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %143 = "tosa.mul"(%141, %142) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %144 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %145 = "tosa.add"(%143, %144) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %146 = "tosa.clamp"(%145) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
    %147 = "tosa.depthwise_conv2d"(%146, %27, %13) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x144xf32>, tensor<3x3x144x1xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %148 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %149 = "tosa.sub"(%147, %148) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %150 = "tosa.add"(%28, %1) : (tensor<144xf32>, tensor<1xf32>) -> tensor<144xf32>
    %151 = "tosa.rsqrt"(%150) : (tensor<144xf32>) -> tensor<144xf32>
    %152 = "tosa.reshape"(%151) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %153 = "tosa.mul"(%149, %152) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %154 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %155 = "tosa.mul"(%153, %154) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %156 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %157 = "tosa.add"(%155, %156) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %158 = "tosa.clamp"(%157) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
    %159 = "tosa.transpose"(%24, %2) : (tensor<1x1x144x24xf32>, tensor<4xi32>) -> tensor<24x1x1x144xf32>
    %160 = "tosa.conv2d"(%158, %159, %14) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x56x56x144xf32>, tensor<24x1x1x144xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %161 = "tosa.reshape"(%25) {new_shape = array<i64: 1, 1, 1, 24>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %162 = "tosa.sub"(%160, %161) : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %163 = "tosa.add"(%25, %1) : (tensor<24xf32>, tensor<1xf32>) -> tensor<24xf32>
    %164 = "tosa.rsqrt"(%163) : (tensor<24xf32>) -> tensor<24xf32>
    %165 = "tosa.reshape"(%164) {new_shape = array<i64: 1, 1, 1, 24>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %166 = "tosa.mul"(%162, %165) {shift = 0 : i32} : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %167 = "tosa.reshape"(%25) {new_shape = array<i64: 1, 1, 1, 24>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %168 = "tosa.mul"(%166, %167) {shift = 0 : i32} : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %169 = "tosa.reshape"(%25) {new_shape = array<i64: 1, 1, 1, 24>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %170 = "tosa.add"(%168, %169) : (tensor<1x56x56x24xf32>, tensor<1x1x1x24xf32>) -> tensor<1x56x56x24xf32>
    %171 = "tosa.add"(%133, %170) : (tensor<1x56x56x24xf32>, tensor<1x56x56x24xf32>) -> tensor<1x56x56x24xf32>
    %172 = "tosa.transpose"(%26, %2) : (tensor<1x1x24x144xf32>, tensor<4xi32>) -> tensor<144x1x1x24xf32>
    %173 = "tosa.conv2d"(%171, %172, %13) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x56x56x24xf32>, tensor<144x1x1x24xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %174 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %175 = "tosa.sub"(%173, %174) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %176 = "tosa.add"(%28, %1) : (tensor<144xf32>, tensor<1xf32>) -> tensor<144xf32>
    %177 = "tosa.rsqrt"(%176) : (tensor<144xf32>) -> tensor<144xf32>
    %178 = "tosa.reshape"(%177) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %179 = "tosa.mul"(%175, %178) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %180 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %181 = "tosa.mul"(%179, %180) {shift = 0 : i32} : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %182 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %183 = "tosa.add"(%181, %182) : (tensor<1x56x56x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x56x56x144xf32>
    %184 = "tosa.clamp"(%183) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x56x56x144xf32>) -> tensor<1x56x56x144xf32>
    %185 = "tosa.pad"(%184, %16) : (tensor<1x56x56x144xf32>, tensor<4x2xi32>) -> tensor<1x57x57x144xf32>
    %186 = "tosa.depthwise_conv2d"(%185, %27, %13) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x57x57x144xf32>, tensor<3x3x144x1xf32>, tensor<144xf32>) -> tensor<1x28x28x144xf32>
    %187 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %188 = "tosa.sub"(%186, %187) : (tensor<1x28x28x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x28x28x144xf32>
    %189 = "tosa.add"(%28, %1) : (tensor<144xf32>, tensor<1xf32>) -> tensor<144xf32>
    %190 = "tosa.rsqrt"(%189) : (tensor<144xf32>) -> tensor<144xf32>
    %191 = "tosa.reshape"(%190) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %192 = "tosa.mul"(%188, %191) {shift = 0 : i32} : (tensor<1x28x28x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x28x28x144xf32>
    %193 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %194 = "tosa.mul"(%192, %193) {shift = 0 : i32} : (tensor<1x28x28x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x28x28x144xf32>
    %195 = "tosa.reshape"(%28) {new_shape = array<i64: 1, 1, 1, 144>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %196 = "tosa.add"(%194, %195) : (tensor<1x28x28x144xf32>, tensor<1x1x1x144xf32>) -> tensor<1x28x28x144xf32>
    %197 = "tosa.clamp"(%196) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
    %198 = "tosa.transpose"(%29, %2) : (tensor<1x1x144x32xf32>, tensor<4xi32>) -> tensor<32x1x1x144xf32>
    %199 = "tosa.conv2d"(%197, %198, %12) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x144xf32>, tensor<32x1x1x144xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %200 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %201 = "tosa.sub"(%199, %200) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %202 = "tosa.add"(%31, %1) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %203 = "tosa.rsqrt"(%202) : (tensor<32xf32>) -> tensor<32xf32>
    %204 = "tosa.reshape"(%203) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %205 = "tosa.mul"(%201, %204) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %206 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %207 = "tosa.mul"(%205, %206) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %208 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %209 = "tosa.add"(%207, %208) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %210 = "tosa.transpose"(%32, %2) : (tensor<1x1x32x192xf32>, tensor<4xi32>) -> tensor<192x1x1x32xf32>
    %211 = "tosa.conv2d"(%209, %210, %11) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x32xf32>, tensor<192x1x1x32xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %212 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %213 = "tosa.sub"(%211, %212) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %214 = "tosa.add"(%34, %1) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %215 = "tosa.rsqrt"(%214) : (tensor<192xf32>) -> tensor<192xf32>
    %216 = "tosa.reshape"(%215) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %217 = "tosa.mul"(%213, %216) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %218 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %219 = "tosa.mul"(%217, %218) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %220 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %221 = "tosa.add"(%219, %220) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %222 = "tosa.clamp"(%221) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %223 = "tosa.depthwise_conv2d"(%222, %33, %11) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x192xf32>, tensor<3x3x192x1xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %224 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %225 = "tosa.sub"(%223, %224) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %226 = "tosa.add"(%34, %1) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %227 = "tosa.rsqrt"(%226) : (tensor<192xf32>) -> tensor<192xf32>
    %228 = "tosa.reshape"(%227) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %229 = "tosa.mul"(%225, %228) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %230 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %231 = "tosa.mul"(%229, %230) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %232 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %233 = "tosa.add"(%231, %232) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %234 = "tosa.clamp"(%233) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %235 = "tosa.transpose"(%30, %2) : (tensor<1x1x192x32xf32>, tensor<4xi32>) -> tensor<32x1x1x192xf32>
    %236 = "tosa.conv2d"(%234, %235, %12) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x192xf32>, tensor<32x1x1x192xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %237 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %238 = "tosa.sub"(%236, %237) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %239 = "tosa.add"(%31, %1) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %240 = "tosa.rsqrt"(%239) : (tensor<32xf32>) -> tensor<32xf32>
    %241 = "tosa.reshape"(%240) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %242 = "tosa.mul"(%238, %241) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %243 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %244 = "tosa.mul"(%242, %243) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %245 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %246 = "tosa.add"(%244, %245) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %247 = "tosa.add"(%209, %246) : (tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
    %248 = "tosa.transpose"(%32, %2) : (tensor<1x1x32x192xf32>, tensor<4xi32>) -> tensor<192x1x1x32xf32>
    %249 = "tosa.conv2d"(%247, %248, %11) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x32xf32>, tensor<192x1x1x32xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %250 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %251 = "tosa.sub"(%249, %250) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %252 = "tosa.add"(%34, %1) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %253 = "tosa.rsqrt"(%252) : (tensor<192xf32>) -> tensor<192xf32>
    %254 = "tosa.reshape"(%253) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %255 = "tosa.mul"(%251, %254) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %256 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %257 = "tosa.mul"(%255, %256) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %258 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %259 = "tosa.add"(%257, %258) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %260 = "tosa.clamp"(%259) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %261 = "tosa.depthwise_conv2d"(%260, %33, %11) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x28x28x192xf32>, tensor<3x3x192x1xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %262 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %263 = "tosa.sub"(%261, %262) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %264 = "tosa.add"(%34, %1) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %265 = "tosa.rsqrt"(%264) : (tensor<192xf32>) -> tensor<192xf32>
    %266 = "tosa.reshape"(%265) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %267 = "tosa.mul"(%263, %266) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %268 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %269 = "tosa.mul"(%267, %268) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %270 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %271 = "tosa.add"(%269, %270) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %272 = "tosa.clamp"(%271) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %273 = "tosa.transpose"(%30, %2) : (tensor<1x1x192x32xf32>, tensor<4xi32>) -> tensor<32x1x1x192xf32>
    %274 = "tosa.conv2d"(%272, %273, %12) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x192xf32>, tensor<32x1x1x192xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %275 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %276 = "tosa.sub"(%274, %275) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %277 = "tosa.add"(%31, %1) : (tensor<32xf32>, tensor<1xf32>) -> tensor<32xf32>
    %278 = "tosa.rsqrt"(%277) : (tensor<32xf32>) -> tensor<32xf32>
    %279 = "tosa.reshape"(%278) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %280 = "tosa.mul"(%276, %279) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %281 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %282 = "tosa.mul"(%280, %281) {shift = 0 : i32} : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %283 = "tosa.reshape"(%31) {new_shape = array<i64: 1, 1, 1, 32>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %284 = "tosa.add"(%282, %283) : (tensor<1x28x28x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x28x28x32xf32>
    %285 = "tosa.add"(%247, %284) : (tensor<1x28x28x32xf32>, tensor<1x28x28x32xf32>) -> tensor<1x28x28x32xf32>
    %286 = "tosa.transpose"(%32, %2) : (tensor<1x1x32x192xf32>, tensor<4xi32>) -> tensor<192x1x1x32xf32>
    %287 = "tosa.conv2d"(%285, %286, %11) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x32xf32>, tensor<192x1x1x32xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %288 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %289 = "tosa.sub"(%287, %288) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %290 = "tosa.add"(%34, %1) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %291 = "tosa.rsqrt"(%290) : (tensor<192xf32>) -> tensor<192xf32>
    %292 = "tosa.reshape"(%291) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %293 = "tosa.mul"(%289, %292) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %294 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %295 = "tosa.mul"(%293, %294) {shift = 0 : i32} : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %296 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %297 = "tosa.add"(%295, %296) : (tensor<1x28x28x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x28x28x192xf32>
    %298 = "tosa.clamp"(%297) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x28x28x192xf32>) -> tensor<1x28x28x192xf32>
    %299 = "tosa.pad"(%298, %16) : (tensor<1x28x28x192xf32>, tensor<4x2xi32>) -> tensor<1x29x29x192xf32>
    %300 = "tosa.depthwise_conv2d"(%299, %33, %11) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x29x29x192xf32>, tensor<3x3x192x1xf32>, tensor<192xf32>) -> tensor<1x14x14x192xf32>
    %301 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %302 = "tosa.sub"(%300, %301) : (tensor<1x14x14x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x14x14x192xf32>
    %303 = "tosa.add"(%34, %1) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
    %304 = "tosa.rsqrt"(%303) : (tensor<192xf32>) -> tensor<192xf32>
    %305 = "tosa.reshape"(%304) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %306 = "tosa.mul"(%302, %305) {shift = 0 : i32} : (tensor<1x14x14x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x14x14x192xf32>
    %307 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %308 = "tosa.mul"(%306, %307) {shift = 0 : i32} : (tensor<1x14x14x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x14x14x192xf32>
    %309 = "tosa.reshape"(%34) {new_shape = array<i64: 1, 1, 1, 192>} : (tensor<192xf32>) -> tensor<1x1x1x192xf32>
    %310 = "tosa.add"(%308, %309) : (tensor<1x14x14x192xf32>, tensor<1x1x1x192xf32>) -> tensor<1x14x14x192xf32>
    %311 = "tosa.clamp"(%310) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x192xf32>) -> tensor<1x14x14x192xf32>
    %312 = "tosa.transpose"(%35, %2) : (tensor<1x1x192x64xf32>, tensor<4xi32>) -> tensor<64x1x1x192xf32>
    %313 = "tosa.conv2d"(%311, %312, %10) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x192xf32>, tensor<64x1x1x192xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %314 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %315 = "tosa.sub"(%313, %314) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %316 = "tosa.add"(%37, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %317 = "tosa.rsqrt"(%316) : (tensor<64xf32>) -> tensor<64xf32>
    %318 = "tosa.reshape"(%317) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %319 = "tosa.mul"(%315, %318) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %320 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %321 = "tosa.mul"(%319, %320) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %322 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %323 = "tosa.add"(%321, %322) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %324 = "tosa.transpose"(%38, %2) : (tensor<1x1x64x384xf32>, tensor<4xi32>) -> tensor<384x1x1x64xf32>
    %325 = "tosa.conv2d"(%323, %324, %9) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x64xf32>, tensor<384x1x1x64xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %326 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %327 = "tosa.sub"(%325, %326) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %328 = "tosa.add"(%40, %1) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %329 = "tosa.rsqrt"(%328) : (tensor<384xf32>) -> tensor<384xf32>
    %330 = "tosa.reshape"(%329) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %331 = "tosa.mul"(%327, %330) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %332 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %333 = "tosa.mul"(%331, %332) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %334 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %335 = "tosa.add"(%333, %334) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %336 = "tosa.clamp"(%335) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %337 = "tosa.depthwise_conv2d"(%336, %39, %9) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x384xf32>, tensor<3x3x384x1xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %338 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %339 = "tosa.sub"(%337, %338) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %340 = "tosa.add"(%40, %1) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %341 = "tosa.rsqrt"(%340) : (tensor<384xf32>) -> tensor<384xf32>
    %342 = "tosa.reshape"(%341) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %343 = "tosa.mul"(%339, %342) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %344 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %345 = "tosa.mul"(%343, %344) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %346 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %347 = "tosa.add"(%345, %346) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %348 = "tosa.clamp"(%347) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %349 = "tosa.transpose"(%36, %2) : (tensor<1x1x384x64xf32>, tensor<4xi32>) -> tensor<64x1x1x384xf32>
    %350 = "tosa.conv2d"(%348, %349, %10) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x384xf32>, tensor<64x1x1x384xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %351 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %352 = "tosa.sub"(%350, %351) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %353 = "tosa.add"(%37, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %354 = "tosa.rsqrt"(%353) : (tensor<64xf32>) -> tensor<64xf32>
    %355 = "tosa.reshape"(%354) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %356 = "tosa.mul"(%352, %355) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %357 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %358 = "tosa.mul"(%356, %357) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %359 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %360 = "tosa.add"(%358, %359) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %361 = "tosa.add"(%323, %360) : (tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
    %362 = "tosa.transpose"(%38, %2) : (tensor<1x1x64x384xf32>, tensor<4xi32>) -> tensor<384x1x1x64xf32>
    %363 = "tosa.conv2d"(%361, %362, %9) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x64xf32>, tensor<384x1x1x64xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %364 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %365 = "tosa.sub"(%363, %364) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %366 = "tosa.add"(%40, %1) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %367 = "tosa.rsqrt"(%366) : (tensor<384xf32>) -> tensor<384xf32>
    %368 = "tosa.reshape"(%367) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %369 = "tosa.mul"(%365, %368) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %370 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %371 = "tosa.mul"(%369, %370) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %372 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %373 = "tosa.add"(%371, %372) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %374 = "tosa.clamp"(%373) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %375 = "tosa.depthwise_conv2d"(%374, %39, %9) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x384xf32>, tensor<3x3x384x1xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %376 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %377 = "tosa.sub"(%375, %376) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %378 = "tosa.add"(%40, %1) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %379 = "tosa.rsqrt"(%378) : (tensor<384xf32>) -> tensor<384xf32>
    %380 = "tosa.reshape"(%379) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %381 = "tosa.mul"(%377, %380) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %382 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %383 = "tosa.mul"(%381, %382) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %384 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %385 = "tosa.add"(%383, %384) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %386 = "tosa.clamp"(%385) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %387 = "tosa.transpose"(%36, %2) : (tensor<1x1x384x64xf32>, tensor<4xi32>) -> tensor<64x1x1x384xf32>
    %388 = "tosa.conv2d"(%386, %387, %10) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x384xf32>, tensor<64x1x1x384xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %389 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %390 = "tosa.sub"(%388, %389) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %391 = "tosa.add"(%37, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %392 = "tosa.rsqrt"(%391) : (tensor<64xf32>) -> tensor<64xf32>
    %393 = "tosa.reshape"(%392) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %394 = "tosa.mul"(%390, %393) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %395 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %396 = "tosa.mul"(%394, %395) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %397 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %398 = "tosa.add"(%396, %397) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %399 = "tosa.add"(%361, %398) : (tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
    %400 = "tosa.transpose"(%38, %2) : (tensor<1x1x64x384xf32>, tensor<4xi32>) -> tensor<384x1x1x64xf32>
    %401 = "tosa.conv2d"(%399, %400, %9) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x64xf32>, tensor<384x1x1x64xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %402 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %403 = "tosa.sub"(%401, %402) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %404 = "tosa.add"(%40, %1) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %405 = "tosa.rsqrt"(%404) : (tensor<384xf32>) -> tensor<384xf32>
    %406 = "tosa.reshape"(%405) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %407 = "tosa.mul"(%403, %406) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %408 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %409 = "tosa.mul"(%407, %408) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %410 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %411 = "tosa.add"(%409, %410) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %412 = "tosa.clamp"(%411) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %413 = "tosa.depthwise_conv2d"(%412, %39, %9) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x384xf32>, tensor<3x3x384x1xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %414 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %415 = "tosa.sub"(%413, %414) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %416 = "tosa.add"(%40, %1) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %417 = "tosa.rsqrt"(%416) : (tensor<384xf32>) -> tensor<384xf32>
    %418 = "tosa.reshape"(%417) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %419 = "tosa.mul"(%415, %418) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %420 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %421 = "tosa.mul"(%419, %420) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %422 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %423 = "tosa.add"(%421, %422) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %424 = "tosa.clamp"(%423) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %425 = "tosa.transpose"(%36, %2) : (tensor<1x1x384x64xf32>, tensor<4xi32>) -> tensor<64x1x1x384xf32>
    %426 = "tosa.conv2d"(%424, %425, %10) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x384xf32>, tensor<64x1x1x384xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %427 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %428 = "tosa.sub"(%426, %427) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %429 = "tosa.add"(%37, %1) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %430 = "tosa.rsqrt"(%429) : (tensor<64xf32>) -> tensor<64xf32>
    %431 = "tosa.reshape"(%430) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %432 = "tosa.mul"(%428, %431) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %433 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %434 = "tosa.mul"(%432, %433) {shift = 0 : i32} : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %435 = "tosa.reshape"(%37) {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %436 = "tosa.add"(%434, %435) : (tensor<1x14x14x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x14x14x64xf32>
    %437 = "tosa.add"(%399, %436) : (tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
    %438 = "tosa.transpose"(%38, %2) : (tensor<1x1x64x384xf32>, tensor<4xi32>) -> tensor<384x1x1x64xf32>
    %439 = "tosa.conv2d"(%437, %438, %9) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x64xf32>, tensor<384x1x1x64xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %440 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %441 = "tosa.sub"(%439, %440) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %442 = "tosa.add"(%40, %1) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %443 = "tosa.rsqrt"(%442) : (tensor<384xf32>) -> tensor<384xf32>
    %444 = "tosa.reshape"(%443) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %445 = "tosa.mul"(%441, %444) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %446 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %447 = "tosa.mul"(%445, %446) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %448 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %449 = "tosa.add"(%447, %448) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %450 = "tosa.clamp"(%449) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %451 = "tosa.depthwise_conv2d"(%450, %39, %9) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x384xf32>, tensor<3x3x384x1xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %452 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %453 = "tosa.sub"(%451, %452) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %454 = "tosa.add"(%40, %1) : (tensor<384xf32>, tensor<1xf32>) -> tensor<384xf32>
    %455 = "tosa.rsqrt"(%454) : (tensor<384xf32>) -> tensor<384xf32>
    %456 = "tosa.reshape"(%455) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %457 = "tosa.mul"(%453, %456) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %458 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %459 = "tosa.mul"(%457, %458) {shift = 0 : i32} : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %460 = "tosa.reshape"(%40) {new_shape = array<i64: 1, 1, 1, 384>} : (tensor<384xf32>) -> tensor<1x1x1x384xf32>
    %461 = "tosa.add"(%459, %460) : (tensor<1x14x14x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1x14x14x384xf32>
    %462 = "tosa.clamp"(%461) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x384xf32>) -> tensor<1x14x14x384xf32>
    %463 = "tosa.transpose"(%41, %2) : (tensor<1x1x384x96xf32>, tensor<4xi32>) -> tensor<96x1x1x384xf32>
    %464 = "tosa.conv2d"(%462, %463, %8) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x384xf32>, tensor<96x1x1x384xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %465 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %466 = "tosa.sub"(%464, %465) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %467 = "tosa.add"(%43, %1) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %468 = "tosa.rsqrt"(%467) : (tensor<96xf32>) -> tensor<96xf32>
    %469 = "tosa.reshape"(%468) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %470 = "tosa.mul"(%466, %469) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %471 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %472 = "tosa.mul"(%470, %471) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %473 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %474 = "tosa.add"(%472, %473) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %475 = "tosa.transpose"(%44, %2) : (tensor<1x1x96x576xf32>, tensor<4xi32>) -> tensor<576x1x1x96xf32>
    %476 = "tosa.conv2d"(%474, %475, %7) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x96xf32>, tensor<576x1x1x96xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %477 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %478 = "tosa.sub"(%476, %477) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %479 = "tosa.add"(%46, %1) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %480 = "tosa.rsqrt"(%479) : (tensor<576xf32>) -> tensor<576xf32>
    %481 = "tosa.reshape"(%480) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %482 = "tosa.mul"(%478, %481) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %483 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %484 = "tosa.mul"(%482, %483) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %485 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %486 = "tosa.add"(%484, %485) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %487 = "tosa.clamp"(%486) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %488 = "tosa.depthwise_conv2d"(%487, %45, %7) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x576xf32>, tensor<3x3x576x1xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %489 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %490 = "tosa.sub"(%488, %489) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %491 = "tosa.add"(%46, %1) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %492 = "tosa.rsqrt"(%491) : (tensor<576xf32>) -> tensor<576xf32>
    %493 = "tosa.reshape"(%492) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %494 = "tosa.mul"(%490, %493) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %495 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %496 = "tosa.mul"(%494, %495) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %497 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %498 = "tosa.add"(%496, %497) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %499 = "tosa.clamp"(%498) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %500 = "tosa.transpose"(%42, %2) : (tensor<1x1x576x96xf32>, tensor<4xi32>) -> tensor<96x1x1x576xf32>
    %501 = "tosa.conv2d"(%499, %500, %8) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x576xf32>, tensor<96x1x1x576xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %502 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %503 = "tosa.sub"(%501, %502) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %504 = "tosa.add"(%43, %1) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %505 = "tosa.rsqrt"(%504) : (tensor<96xf32>) -> tensor<96xf32>
    %506 = "tosa.reshape"(%505) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %507 = "tosa.mul"(%503, %506) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %508 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %509 = "tosa.mul"(%507, %508) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %510 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %511 = "tosa.add"(%509, %510) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %512 = "tosa.add"(%474, %511) : (tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
    %513 = "tosa.transpose"(%44, %2) : (tensor<1x1x96x576xf32>, tensor<4xi32>) -> tensor<576x1x1x96xf32>
    %514 = "tosa.conv2d"(%512, %513, %7) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x96xf32>, tensor<576x1x1x96xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %515 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %516 = "tosa.sub"(%514, %515) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %517 = "tosa.add"(%46, %1) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %518 = "tosa.rsqrt"(%517) : (tensor<576xf32>) -> tensor<576xf32>
    %519 = "tosa.reshape"(%518) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %520 = "tosa.mul"(%516, %519) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %521 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %522 = "tosa.mul"(%520, %521) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %523 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %524 = "tosa.add"(%522, %523) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %525 = "tosa.clamp"(%524) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %526 = "tosa.depthwise_conv2d"(%525, %45, %7) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x576xf32>, tensor<3x3x576x1xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %527 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %528 = "tosa.sub"(%526, %527) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %529 = "tosa.add"(%46, %1) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %530 = "tosa.rsqrt"(%529) : (tensor<576xf32>) -> tensor<576xf32>
    %531 = "tosa.reshape"(%530) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %532 = "tosa.mul"(%528, %531) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %533 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %534 = "tosa.mul"(%532, %533) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %535 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %536 = "tosa.add"(%534, %535) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %537 = "tosa.clamp"(%536) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %538 = "tosa.transpose"(%42, %2) : (tensor<1x1x576x96xf32>, tensor<4xi32>) -> tensor<96x1x1x576xf32>
    %539 = "tosa.conv2d"(%537, %538, %8) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x576xf32>, tensor<96x1x1x576xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %540 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %541 = "tosa.sub"(%539, %540) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %542 = "tosa.add"(%43, %1) : (tensor<96xf32>, tensor<1xf32>) -> tensor<96xf32>
    %543 = "tosa.rsqrt"(%542) : (tensor<96xf32>) -> tensor<96xf32>
    %544 = "tosa.reshape"(%543) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %545 = "tosa.mul"(%541, %544) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %546 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %547 = "tosa.mul"(%545, %546) {shift = 0 : i32} : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %548 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 1, 1, 96>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %549 = "tosa.add"(%547, %548) : (tensor<1x14x14x96xf32>, tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %550 = "tosa.add"(%512, %549) : (tensor<1x14x14x96xf32>, tensor<1x14x14x96xf32>) -> tensor<1x14x14x96xf32>
    %551 = "tosa.transpose"(%44, %2) : (tensor<1x1x96x576xf32>, tensor<4xi32>) -> tensor<576x1x1x96xf32>
    %552 = "tosa.conv2d"(%550, %551, %7) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x14x14x96xf32>, tensor<576x1x1x96xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %553 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %554 = "tosa.sub"(%552, %553) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %555 = "tosa.add"(%46, %1) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %556 = "tosa.rsqrt"(%555) : (tensor<576xf32>) -> tensor<576xf32>
    %557 = "tosa.reshape"(%556) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %558 = "tosa.mul"(%554, %557) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %559 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %560 = "tosa.mul"(%558, %559) {shift = 0 : i32} : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %561 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %562 = "tosa.add"(%560, %561) : (tensor<1x14x14x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x14x14x576xf32>
    %563 = "tosa.clamp"(%562) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x14x14x576xf32>) -> tensor<1x14x14x576xf32>
    %564 = "tosa.pad"(%563, %16) : (tensor<1x14x14x576xf32>, tensor<4x2xi32>) -> tensor<1x15x15x576xf32>
    %565 = "tosa.depthwise_conv2d"(%564, %45, %7) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x15x15x576xf32>, tensor<3x3x576x1xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %566 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %567 = "tosa.sub"(%565, %566) : (tensor<1x7x7x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %568 = "tosa.add"(%46, %1) : (tensor<576xf32>, tensor<1xf32>) -> tensor<576xf32>
    %569 = "tosa.rsqrt"(%568) : (tensor<576xf32>) -> tensor<576xf32>
    %570 = "tosa.reshape"(%569) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %571 = "tosa.mul"(%567, %570) {shift = 0 : i32} : (tensor<1x7x7x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %572 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %573 = "tosa.mul"(%571, %572) {shift = 0 : i32} : (tensor<1x7x7x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %574 = "tosa.reshape"(%46) {new_shape = array<i64: 1, 1, 1, 576>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %575 = "tosa.add"(%573, %574) : (tensor<1x7x7x576xf32>, tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %576 = "tosa.clamp"(%575) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x7x7x576xf32>) -> tensor<1x7x7x576xf32>
    %577 = "tosa.transpose"(%47, %2) : (tensor<1x1x576x160xf32>, tensor<4xi32>) -> tensor<160x1x1x576xf32>
    %578 = "tosa.conv2d"(%576, %577, %6) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x576xf32>, tensor<160x1x1x576xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %579 = "tosa.reshape"(%49) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %580 = "tosa.sub"(%578, %579) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %581 = "tosa.add"(%49, %1) : (tensor<160xf32>, tensor<1xf32>) -> tensor<160xf32>
    %582 = "tosa.rsqrt"(%581) : (tensor<160xf32>) -> tensor<160xf32>
    %583 = "tosa.reshape"(%582) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %584 = "tosa.mul"(%580, %583) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %585 = "tosa.reshape"(%49) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %586 = "tosa.mul"(%584, %585) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %587 = "tosa.reshape"(%49) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %588 = "tosa.add"(%586, %587) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %589 = "tosa.transpose"(%50, %2) : (tensor<1x1x160x960xf32>, tensor<4xi32>) -> tensor<960x1x1x160xf32>
    %590 = "tosa.conv2d"(%588, %589, %5) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x160xf32>, tensor<960x1x1x160xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %591 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %592 = "tosa.sub"(%590, %591) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %593 = "tosa.add"(%52, %1) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %594 = "tosa.rsqrt"(%593) : (tensor<960xf32>) -> tensor<960xf32>
    %595 = "tosa.reshape"(%594) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %596 = "tosa.mul"(%592, %595) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %597 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %598 = "tosa.mul"(%596, %597) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %599 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %600 = "tosa.add"(%598, %599) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %601 = "tosa.clamp"(%600) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %602 = "tosa.depthwise_conv2d"(%601, %51, %5) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x960xf32>, tensor<3x3x960x1xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %603 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %604 = "tosa.sub"(%602, %603) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %605 = "tosa.add"(%52, %1) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %606 = "tosa.rsqrt"(%605) : (tensor<960xf32>) -> tensor<960xf32>
    %607 = "tosa.reshape"(%606) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %608 = "tosa.mul"(%604, %607) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %609 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %610 = "tosa.mul"(%608, %609) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %611 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %612 = "tosa.add"(%610, %611) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %613 = "tosa.clamp"(%612) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %614 = "tosa.transpose"(%48, %2) : (tensor<1x1x960x160xf32>, tensor<4xi32>) -> tensor<160x1x1x960xf32>
    %615 = "tosa.conv2d"(%613, %614, %6) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x960xf32>, tensor<160x1x1x960xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %616 = "tosa.reshape"(%49) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %617 = "tosa.sub"(%615, %616) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %618 = "tosa.add"(%49, %1) : (tensor<160xf32>, tensor<1xf32>) -> tensor<160xf32>
    %619 = "tosa.rsqrt"(%618) : (tensor<160xf32>) -> tensor<160xf32>
    %620 = "tosa.reshape"(%619) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %621 = "tosa.mul"(%617, %620) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %622 = "tosa.reshape"(%49) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %623 = "tosa.mul"(%621, %622) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %624 = "tosa.reshape"(%49) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %625 = "tosa.add"(%623, %624) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %626 = "tosa.add"(%588, %625) : (tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
    %627 = "tosa.transpose"(%50, %2) : (tensor<1x1x160x960xf32>, tensor<4xi32>) -> tensor<960x1x1x160xf32>
    %628 = "tosa.conv2d"(%626, %627, %5) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x160xf32>, tensor<960x1x1x160xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %629 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %630 = "tosa.sub"(%628, %629) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %631 = "tosa.add"(%52, %1) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %632 = "tosa.rsqrt"(%631) : (tensor<960xf32>) -> tensor<960xf32>
    %633 = "tosa.reshape"(%632) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %634 = "tosa.mul"(%630, %633) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %635 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %636 = "tosa.mul"(%634, %635) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %637 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %638 = "tosa.add"(%636, %637) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %639 = "tosa.clamp"(%638) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %640 = "tosa.depthwise_conv2d"(%639, %51, %5) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x960xf32>, tensor<3x3x960x1xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %641 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %642 = "tosa.sub"(%640, %641) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %643 = "tosa.add"(%52, %1) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %644 = "tosa.rsqrt"(%643) : (tensor<960xf32>) -> tensor<960xf32>
    %645 = "tosa.reshape"(%644) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %646 = "tosa.mul"(%642, %645) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %647 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %648 = "tosa.mul"(%646, %647) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %649 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %650 = "tosa.add"(%648, %649) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %651 = "tosa.clamp"(%650) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %652 = "tosa.transpose"(%48, %2) : (tensor<1x1x960x160xf32>, tensor<4xi32>) -> tensor<160x1x1x960xf32>
    %653 = "tosa.conv2d"(%651, %652, %6) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x960xf32>, tensor<160x1x1x960xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %654 = "tosa.reshape"(%49) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %655 = "tosa.sub"(%653, %654) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %656 = "tosa.add"(%49, %1) : (tensor<160xf32>, tensor<1xf32>) -> tensor<160xf32>
    %657 = "tosa.rsqrt"(%656) : (tensor<160xf32>) -> tensor<160xf32>
    %658 = "tosa.reshape"(%657) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %659 = "tosa.mul"(%655, %658) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %660 = "tosa.reshape"(%49) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %661 = "tosa.mul"(%659, %660) {shift = 0 : i32} : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %662 = "tosa.reshape"(%49) {new_shape = array<i64: 1, 1, 1, 160>} : (tensor<160xf32>) -> tensor<1x1x1x160xf32>
    %663 = "tosa.add"(%661, %662) : (tensor<1x7x7x160xf32>, tensor<1x1x1x160xf32>) -> tensor<1x7x7x160xf32>
    %664 = "tosa.add"(%626, %663) : (tensor<1x7x7x160xf32>, tensor<1x7x7x160xf32>) -> tensor<1x7x7x160xf32>
    %665 = "tosa.transpose"(%50, %2) : (tensor<1x1x160x960xf32>, tensor<4xi32>) -> tensor<960x1x1x160xf32>
    %666 = "tosa.conv2d"(%664, %665, %5) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x160xf32>, tensor<960x1x1x160xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %667 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %668 = "tosa.sub"(%666, %667) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %669 = "tosa.add"(%52, %1) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %670 = "tosa.rsqrt"(%669) : (tensor<960xf32>) -> tensor<960xf32>
    %671 = "tosa.reshape"(%670) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %672 = "tosa.mul"(%668, %671) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %673 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %674 = "tosa.mul"(%672, %673) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %675 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %676 = "tosa.add"(%674, %675) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %677 = "tosa.clamp"(%676) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %678 = "tosa.depthwise_conv2d"(%677, %51, %5) {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x960xf32>, tensor<3x3x960x1xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %679 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %680 = "tosa.sub"(%678, %679) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %681 = "tosa.add"(%52, %1) : (tensor<960xf32>, tensor<1xf32>) -> tensor<960xf32>
    %682 = "tosa.rsqrt"(%681) : (tensor<960xf32>) -> tensor<960xf32>
    %683 = "tosa.reshape"(%682) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %684 = "tosa.mul"(%680, %683) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %685 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %686 = "tosa.mul"(%684, %685) {shift = 0 : i32} : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %687 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 1, 1, 960>} : (tensor<960xf32>) -> tensor<1x1x1x960xf32>
    %688 = "tosa.add"(%686, %687) : (tensor<1x7x7x960xf32>, tensor<1x1x1x960xf32>) -> tensor<1x7x7x960xf32>
    %689 = "tosa.clamp"(%688) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x7x7x960xf32>) -> tensor<1x7x7x960xf32>
    %690 = "tosa.transpose"(%53, %2) : (tensor<1x1x960x320xf32>, tensor<4xi32>) -> tensor<320x1x1x960xf32>
    %691 = "tosa.conv2d"(%689, %690, %4) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x960xf32>, tensor<320x1x1x960xf32>, tensor<320xf32>) -> tensor<1x7x7x320xf32>
    %692 = "tosa.reshape"(%54) {new_shape = array<i64: 1, 1, 1, 320>} : (tensor<320xf32>) -> tensor<1x1x1x320xf32>
    %693 = "tosa.sub"(%691, %692) : (tensor<1x7x7x320xf32>, tensor<1x1x1x320xf32>) -> tensor<1x7x7x320xf32>
    %694 = "tosa.add"(%54, %1) : (tensor<320xf32>, tensor<1xf32>) -> tensor<320xf32>
    %695 = "tosa.rsqrt"(%694) : (tensor<320xf32>) -> tensor<320xf32>
    %696 = "tosa.reshape"(%695) {new_shape = array<i64: 1, 1, 1, 320>} : (tensor<320xf32>) -> tensor<1x1x1x320xf32>
    %697 = "tosa.mul"(%693, %696) {shift = 0 : i32} : (tensor<1x7x7x320xf32>, tensor<1x1x1x320xf32>) -> tensor<1x7x7x320xf32>
    %698 = "tosa.reshape"(%54) {new_shape = array<i64: 1, 1, 1, 320>} : (tensor<320xf32>) -> tensor<1x1x1x320xf32>
    %699 = "tosa.mul"(%697, %698) {shift = 0 : i32} : (tensor<1x7x7x320xf32>, tensor<1x1x1x320xf32>) -> tensor<1x7x7x320xf32>
    %700 = "tosa.reshape"(%54) {new_shape = array<i64: 1, 1, 1, 320>} : (tensor<320xf32>) -> tensor<1x1x1x320xf32>
    %701 = "tosa.add"(%699, %700) : (tensor<1x7x7x320xf32>, tensor<1x1x1x320xf32>) -> tensor<1x7x7x320xf32>
    %702 = "tosa.transpose"(%55, %2) : (tensor<1x1x320x1280xf32>, tensor<4xi32>) -> tensor<1280x1x1x320xf32>
    %703 = "tosa.conv2d"(%701, %702, %3) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x320xf32>, tensor<1280x1x1x320xf32>, tensor<1280xf32>) -> tensor<1x7x7x1280xf32>
    %704 = "tosa.reshape"(%56) {new_shape = array<i64: 1, 1, 1, 1280>} : (tensor<1280xf32>) -> tensor<1x1x1x1280xf32>
    %705 = "tosa.sub"(%703, %704) : (tensor<1x7x7x1280xf32>, tensor<1x1x1x1280xf32>) -> tensor<1x7x7x1280xf32>
    %706 = "tosa.add"(%56, %1) : (tensor<1280xf32>, tensor<1xf32>) -> tensor<1280xf32>
    %707 = "tosa.rsqrt"(%706) : (tensor<1280xf32>) -> tensor<1280xf32>
    %708 = "tosa.reshape"(%707) {new_shape = array<i64: 1, 1, 1, 1280>} : (tensor<1280xf32>) -> tensor<1x1x1x1280xf32>
    %709 = "tosa.mul"(%705, %708) {shift = 0 : i32} : (tensor<1x7x7x1280xf32>, tensor<1x1x1x1280xf32>) -> tensor<1x7x7x1280xf32>
    %710 = "tosa.reshape"(%56) {new_shape = array<i64: 1, 1, 1, 1280>} : (tensor<1280xf32>) -> tensor<1x1x1x1280xf32>
    %711 = "tosa.mul"(%709, %710) {shift = 0 : i32} : (tensor<1x7x7x1280xf32>, tensor<1x1x1x1280xf32>) -> tensor<1x7x7x1280xf32>
    %712 = "tosa.reshape"(%56) {new_shape = array<i64: 1, 1, 1, 1280>} : (tensor<1280xf32>) -> tensor<1x1x1x1280xf32>
    %713 = "tosa.add"(%711, %712) : (tensor<1x7x7x1280xf32>, tensor<1x1x1x1280xf32>) -> tensor<1x7x7x1280xf32>
    %714 = "tosa.clamp"(%713) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x7x7x1280xf32>) -> tensor<1x7x7x1280xf32>
    %715 = "tosa.reduce_sum"(%714) {axis = 1 : i32} : (tensor<1x7x7x1280xf32>) -> tensor<1x1x7x1280xf32>
    %716 = "tosa.reduce_sum"(%715) {axis = 2 : i32} : (tensor<1x1x7x1280xf32>) -> tensor<1x1x1x1280xf32>
    %717 = "tosa.reshape"(%716) {new_shape = array<i64: 1, 1280>} : (tensor<1x1x1x1280xf32>) -> tensor<1x1280xf32>
    %718 = "tosa.reshape"(%0) {new_shape = array<i64: 1, 1>} : (tensor<f32>) -> tensor<1x1xf32>
    %719 = "tosa.mul"(%717, %718) {shift = 0 : i32} : (tensor<1x1280xf32>, tensor<1x1xf32>) -> tensor<1x1280xf32>
    %720 = "tosa.reshape"(%719) {new_shape = array<i64: 1, 1, 1280>} : (tensor<1x1280xf32>) -> tensor<1x1x1280xf32>
    %721 = "tosa.reshape"(%57) {new_shape = array<i64: 1, 1280, 1000>} : (tensor<1280x1000xf32>) -> tensor<1x1280x1000xf32>
    %722 = "tosa.matmul"(%720, %721) : (tensor<1x1x1280xf32>, tensor<1x1280x1000xf32>) -> tensor<1x1x1000xf32>
    %723 = "tosa.reshape"(%722) {new_shape = array<i64: 1, 1000>} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
    %724 = "tosa.reshape"(%58) {new_shape = array<i64: 1, 1000>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    %725 = "tosa.add"(%723, %724) : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %726 = "tosa.exp"(%725) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %727 = "tosa.reduce_sum"(%726) {axis = 1 : i32} : (tensor<1x1000xf32>) -> tensor<1x1xf32>
    %728 = "tosa.reciprocal"(%727) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %729 = "tosa.mul"(%726, %728) {shift = 0 : i32} : (tensor<1x1000xf32>, tensor<1x1xf32>) -> tensor<1x1000xf32>
    return %729 : tensor<1x1000xf32>
  }
}

