// RUN: emitc-opt %s --convert-mhlo-region-ops-to-emitc --convert-mhlo-to-emitc | emitc-translate --mlir-to-cpp
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 716 : i32}, tf_saved_model.semantics}  {
  func @predict(%arg0: tensor<1x224x224x3xf32> {tf._user_specified_name = "args_0", tf_saved_model.index_path = [0]}) -> (tensor<1x1000xf32> {tf_saved_model.index_path = []}) attributes {tf._construction_context = "kEagerRuntime"} {
    %0 = mhlo.constant dense<5.000000e-01> : tensor<1x1000xf32>
    %1 = mhlo.constant dense<4.900000e+01> : tensor<1x1280xf32>
    %2 = mhlo.constant dense<5.000000e-01> : tensor<3x3x1x960xf32>
    %3 = mhlo.constant dense<5.000000e-01> : tensor<3x3x1x576xf32>
    %4 = mhlo.constant dense<5.000000e-01> : tensor<3x3x1x384xf32>
    %5 = mhlo.constant dense<5.000000e-01> : tensor<3x3x1x192xf32>
    %6 = mhlo.constant dense<5.000000e-01> : tensor<3x3x1x144xf32>
    %7 = mhlo.constant dense<5.000000e-01> : tensor<3x3x1x96xf32>
    %8 = mhlo.constant dense<5.000000e-01> : tensor<3x3x1x32xf32>
    %9 = mhlo.constant dense<5.000000e-01> : tensor<1280x1000xf32>
    %10 = mhlo.constant dense<5.000000e-01> : tensor<1280xf32>
    %11 = mhlo.constant dense<5.000000e-01> : tensor<1x1x320x1280xf32>
    %12 = mhlo.constant dense<5.000000e-01> : tensor<320xf32>
    %13 = mhlo.constant dense<5.000000e-01> : tensor<1x1x960x320xf32>
    %14 = mhlo.constant dense<5.000000e-01> : tensor<960xf32>
    %15 = mhlo.constant dense<5.000000e-01> : tensor<1x1x160x960xf32>
    %16 = mhlo.constant dense<5.000000e-01> : tensor<160xf32>
    %17 = mhlo.constant dense<5.000000e-01> : tensor<1x1x960x160xf32>
    %18 = mhlo.constant dense<5.000000e-01> : tensor<1x1x576x160xf32>
    %19 = mhlo.constant dense<5.000000e-01> : tensor<576xf32>
    %20 = mhlo.constant dense<5.000000e-01> : tensor<1x1x96x576xf32>
    %21 = mhlo.constant dense<5.000000e-01> : tensor<96xf32>
    %22 = mhlo.constant dense<5.000000e-01> : tensor<1x1x576x96xf32>
    %23 = mhlo.constant dense<5.000000e-01> : tensor<1x1x384x96xf32>
    %24 = mhlo.constant dense<5.000000e-01> : tensor<384xf32>
    %25 = mhlo.constant dense<5.000000e-01> : tensor<1x1x64x384xf32>
    %26 = mhlo.constant dense<5.000000e-01> : tensor<64xf32>
    %27 = mhlo.constant dense<5.000000e-01> : tensor<1x1x384x64xf32>
    %28 = mhlo.constant dense<5.000000e-01> : tensor<1x1x192x64xf32>
    %29 = mhlo.constant dense<5.000000e-01> : tensor<192xf32>
    %30 = mhlo.constant dense<5.000000e-01> : tensor<1x1x32x192xf32>
    %31 = mhlo.constant dense<5.000000e-01> : tensor<32xf32>
    %32 = mhlo.constant dense<5.000000e-01> : tensor<1x1x192x32xf32>
    %33 = mhlo.constant dense<5.000000e-01> : tensor<1x1x144x32xf32>
    %34 = mhlo.constant dense<5.000000e-01> : tensor<144xf32>
    %35 = mhlo.constant dense<5.000000e-01> : tensor<1x1x24x144xf32>
    %36 = mhlo.constant dense<5.000000e-01> : tensor<24xf32>
    %37 = mhlo.constant dense<5.000000e-01> : tensor<1x1x144x24xf32>
    %38 = mhlo.constant dense<5.000000e-01> : tensor<1x1x96x24xf32>
    %39 = mhlo.constant dense<5.000000e-01> : tensor<1x1x16x96xf32>
    %40 = mhlo.constant dense<5.000000e-01> : tensor<16xf32>
    %41 = mhlo.constant dense<5.000000e-01> : tensor<1x1x32x16xf32>
    %42 = mhlo.constant dense<5.000000e-01> : tensor<3x3x3x32xf32>
    %43 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = mhlo.constant dense<6.000000e+00> : tensor<f32>
    %45 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %46 = mhlo.convolution(%arg0, %42) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x224x224x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32>
    %47 = "mhlo.batch_norm_inference"(%46, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %48 = "mhlo.clamp"(%43, %47, %44) : (tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) -> tensor<1x112x112x32xf32>
    %49 = mhlo.convolution(%48, %8) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<1x112x112x32xf32>, tensor<3x3x1x32xf32>) -> tensor<1x112x112x32xf32>
    %50 = "mhlo.batch_norm_inference"(%49, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %51 = "mhlo.clamp"(%43, %50, %44) : (tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) -> tensor<1x112x112x32xf32>
    %52 = mhlo.convolution(%51, %41) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x112x112x32xf32>, tensor<1x1x32x16xf32>) -> tensor<1x112x112x16xf32>
    %53 = "mhlo.batch_norm_inference"(%52, %40, %40, %40, %40) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %54 = mhlo.convolution(%53, %39) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x112x112x16xf32>, tensor<1x1x16x96xf32>) -> tensor<1x112x112x96xf32>
    %55 = "mhlo.batch_norm_inference"(%54, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x112x112x96xf32>
    %56 = "mhlo.clamp"(%43, %55, %44) : (tensor<f32>, tensor<1x112x112x96xf32>, tensor<f32>) -> tensor<1x112x112x96xf32>
    %57 = "mhlo.pad"(%56, %43) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x112x112x96xf32>, tensor<f32>) -> tensor<1x113x113x96xf32>
    %58 = mhlo.convolution(%57, %7) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<1x113x113x96xf32>, tensor<3x3x1x96xf32>) -> tensor<1x56x56x96xf32>
    %59 = "mhlo.batch_norm_inference"(%58, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %60 = "mhlo.clamp"(%43, %59, %44) : (tensor<f32>, tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x96xf32>
    %61 = mhlo.convolution(%60, %38) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x96xf32>, tensor<1x1x96x24xf32>) -> tensor<1x56x56x24xf32>
    %62 = "mhlo.batch_norm_inference"(%61, %36, %36, %36, %36) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %63 = mhlo.convolution(%62, %35) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) -> tensor<1x56x56x144xf32>
    %64 = "mhlo.batch_norm_inference"(%63, %34, %34, %34, %34) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %65 = "mhlo.clamp"(%43, %64, %44) : (tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x56x56x144xf32>
    %66 = mhlo.convolution(%65, %6) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<1x56x56x144xf32>, tensor<3x3x1x144xf32>) -> tensor<1x56x56x144xf32>
    %67 = "mhlo.batch_norm_inference"(%66, %34, %34, %34, %34) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %68 = "mhlo.clamp"(%43, %67, %44) : (tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x56x56x144xf32>
    %69 = mhlo.convolution(%68, %37) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x144xf32>, tensor<1x1x144x24xf32>) -> tensor<1x56x56x24xf32>
    %70 = "mhlo.batch_norm_inference"(%69, %36, %36, %36, %36) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %71 = mhlo.add %62, %70 : tensor<1x56x56x24xf32>
    %72 = mhlo.convolution(%71, %35) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) -> tensor<1x56x56x144xf32>
    %73 = "mhlo.batch_norm_inference"(%72, %34, %34, %34, %34) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %74 = "mhlo.clamp"(%43, %73, %44) : (tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x56x56x144xf32>
    %75 = "mhlo.pad"(%74, %43) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x57x57x144xf32>
    %76 = mhlo.convolution(%75, %6) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<1x57x57x144xf32>, tensor<3x3x1x144xf32>) -> tensor<1x28x28x144xf32>
    %77 = "mhlo.batch_norm_inference"(%76, %34, %34, %34, %34) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x28x28x144xf32>
    %78 = "mhlo.clamp"(%43, %77, %44) : (tensor<f32>, tensor<1x28x28x144xf32>, tensor<f32>) -> tensor<1x28x28x144xf32>
    %79 = mhlo.convolution(%78, %33) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x144xf32>, tensor<1x1x144x32xf32>) -> tensor<1x28x28x32xf32>
    %80 = "mhlo.batch_norm_inference"(%79, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %81 = mhlo.convolution(%80, %30) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) -> tensor<1x28x28x192xf32>
    %82 = "mhlo.batch_norm_inference"(%81, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %83 = "mhlo.clamp"(%43, %82, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %84 = mhlo.convolution(%83, %5) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<1x28x28x192xf32>, tensor<3x3x1x192xf32>) -> tensor<1x28x28x192xf32>
    %85 = "mhlo.batch_norm_inference"(%84, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %86 = "mhlo.clamp"(%43, %85, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %87 = mhlo.convolution(%86, %32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) -> tensor<1x28x28x32xf32>
    %88 = "mhlo.batch_norm_inference"(%87, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %89 = mhlo.add %80, %88 : tensor<1x28x28x32xf32>
    %90 = mhlo.convolution(%89, %30) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) -> tensor<1x28x28x192xf32>
    %91 = "mhlo.batch_norm_inference"(%90, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %92 = "mhlo.clamp"(%43, %91, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %93 = mhlo.convolution(%92, %5) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<1x28x28x192xf32>, tensor<3x3x1x192xf32>) -> tensor<1x28x28x192xf32>
    %94 = "mhlo.batch_norm_inference"(%93, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %95 = "mhlo.clamp"(%43, %94, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %96 = mhlo.convolution(%95, %32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) -> tensor<1x28x28x32xf32>
    %97 = "mhlo.batch_norm_inference"(%96, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %98 = mhlo.add %89, %97 : tensor<1x28x28x32xf32>
    %99 = mhlo.convolution(%98, %30) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) -> tensor<1x28x28x192xf32>
    %100 = "mhlo.batch_norm_inference"(%99, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %101 = "mhlo.clamp"(%43, %100, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %102 = "mhlo.pad"(%101, %43) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x29x29x192xf32>
    %103 = mhlo.convolution(%102, %5) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<1x29x29x192xf32>, tensor<3x3x1x192xf32>) -> tensor<1x14x14x192xf32>
    %104 = "mhlo.batch_norm_inference"(%103, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x14x14x192xf32>
    %105 = "mhlo.clamp"(%43, %104, %44) : (tensor<f32>, tensor<1x14x14x192xf32>, tensor<f32>) -> tensor<1x14x14x192xf32>
    %106 = mhlo.convolution(%105, %28) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x192xf32>, tensor<1x1x192x64xf32>) -> tensor<1x14x14x64xf32>
    %107 = "mhlo.batch_norm_inference"(%106, %26, %26, %26, %26) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %108 = mhlo.convolution(%107, %25) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %109 = "mhlo.batch_norm_inference"(%108, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %110 = "mhlo.clamp"(%43, %109, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %111 = mhlo.convolution(%110, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %112 = "mhlo.batch_norm_inference"(%111, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %113 = "mhlo.clamp"(%43, %112, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %114 = mhlo.convolution(%113, %27) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) -> tensor<1x14x14x64xf32>
    %115 = "mhlo.batch_norm_inference"(%114, %26, %26, %26, %26) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %116 = mhlo.add %107, %115 : tensor<1x14x14x64xf32>
    %117 = mhlo.convolution(%116, %25) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %118 = "mhlo.batch_norm_inference"(%117, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %119 = "mhlo.clamp"(%43, %118, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %120 = mhlo.convolution(%119, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %121 = "mhlo.batch_norm_inference"(%120, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %122 = "mhlo.clamp"(%43, %121, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %123 = mhlo.convolution(%122, %27) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) -> tensor<1x14x14x64xf32>
    %124 = "mhlo.batch_norm_inference"(%123, %26, %26, %26, %26) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %125 = mhlo.add %116, %124 : tensor<1x14x14x64xf32>
    %126 = mhlo.convolution(%125, %25) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %127 = "mhlo.batch_norm_inference"(%126, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %128 = "mhlo.clamp"(%43, %127, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %129 = mhlo.convolution(%128, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %130 = "mhlo.batch_norm_inference"(%129, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %131 = "mhlo.clamp"(%43, %130, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %132 = mhlo.convolution(%131, %27) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) -> tensor<1x14x14x64xf32>
    %133 = "mhlo.batch_norm_inference"(%132, %26, %26, %26, %26) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %134 = mhlo.add %125, %133 : tensor<1x14x14x64xf32>
    %135 = mhlo.convolution(%134, %25) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %136 = "mhlo.batch_norm_inference"(%135, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %137 = "mhlo.clamp"(%43, %136, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %138 = mhlo.convolution(%137, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %139 = "mhlo.batch_norm_inference"(%138, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %140 = "mhlo.clamp"(%43, %139, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %141 = mhlo.convolution(%140, %23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x384xf32>, tensor<1x1x384x96xf32>) -> tensor<1x14x14x96xf32>
    %142 = "mhlo.batch_norm_inference"(%141, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %143 = mhlo.convolution(%142, %20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x14x14x576xf32>
    %144 = "mhlo.batch_norm_inference"(%143, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %145 = "mhlo.clamp"(%43, %144, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %146 = mhlo.convolution(%145, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<1x14x14x576xf32>, tensor<3x3x1x576xf32>) -> tensor<1x14x14x576xf32>
    %147 = "mhlo.batch_norm_inference"(%146, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %148 = "mhlo.clamp"(%43, %147, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %149 = mhlo.convolution(%148, %22) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x14x14x96xf32>
    %150 = "mhlo.batch_norm_inference"(%149, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %151 = mhlo.add %142, %150 : tensor<1x14x14x96xf32>
    %152 = mhlo.convolution(%151, %20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x14x14x576xf32>
    %153 = "mhlo.batch_norm_inference"(%152, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %154 = "mhlo.clamp"(%43, %153, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %155 = mhlo.convolution(%154, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<1x14x14x576xf32>, tensor<3x3x1x576xf32>) -> tensor<1x14x14x576xf32>
    %156 = "mhlo.batch_norm_inference"(%155, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %157 = "mhlo.clamp"(%43, %156, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %158 = mhlo.convolution(%157, %22) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x14x14x96xf32>
    %159 = "mhlo.batch_norm_inference"(%158, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %160 = mhlo.add %151, %159 : tensor<1x14x14x96xf32>
    %161 = mhlo.convolution(%160, %20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x14x14x576xf32>
    %162 = "mhlo.batch_norm_inference"(%161, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %163 = "mhlo.clamp"(%43, %162, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %164 = "mhlo.pad"(%163, %43) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x15x15x576xf32>
    %165 = mhlo.convolution(%164, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<1x15x15x576xf32>, tensor<3x3x1x576xf32>) -> tensor<1x7x7x576xf32>
    %166 = "mhlo.batch_norm_inference"(%165, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %167 = "mhlo.clamp"(%43, %166, %44) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %168 = mhlo.convolution(%167, %18) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x576xf32>, tensor<1x1x576x160xf32>) -> tensor<1x7x7x160xf32>
    %169 = "mhlo.batch_norm_inference"(%168, %16, %16, %16, %16) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %170 = mhlo.convolution(%169, %15) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) -> tensor<1x7x7x960xf32>
    %171 = "mhlo.batch_norm_inference"(%170, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %172 = "mhlo.clamp"(%43, %171, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %173 = mhlo.convolution(%172, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<1x7x7x960xf32>, tensor<3x3x1x960xf32>) -> tensor<1x7x7x960xf32>
    %174 = "mhlo.batch_norm_inference"(%173, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %175 = "mhlo.clamp"(%43, %174, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %176 = mhlo.convolution(%175, %17) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) -> tensor<1x7x7x160xf32>
    %177 = "mhlo.batch_norm_inference"(%176, %16, %16, %16, %16) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %178 = mhlo.add %169, %177 : tensor<1x7x7x160xf32>
    %179 = mhlo.convolution(%178, %15) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) -> tensor<1x7x7x960xf32>
    %180 = "mhlo.batch_norm_inference"(%179, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %181 = "mhlo.clamp"(%43, %180, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %182 = mhlo.convolution(%181, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<1x7x7x960xf32>, tensor<3x3x1x960xf32>) -> tensor<1x7x7x960xf32>
    %183 = "mhlo.batch_norm_inference"(%182, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %184 = "mhlo.clamp"(%43, %183, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %185 = mhlo.convolution(%184, %17) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) -> tensor<1x7x7x160xf32>
    %186 = "mhlo.batch_norm_inference"(%185, %16, %16, %16, %16) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %187 = mhlo.add %178, %186 : tensor<1x7x7x160xf32>
    %188 = mhlo.convolution(%187, %15) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) -> tensor<1x7x7x960xf32>
    %189 = "mhlo.batch_norm_inference"(%188, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %190 = "mhlo.clamp"(%43, %189, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %191 = mhlo.convolution(%190, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<1x7x7x960xf32>, tensor<3x3x1x960xf32>) -> tensor<1x7x7x960xf32>
    %192 = "mhlo.batch_norm_inference"(%191, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %193 = "mhlo.clamp"(%43, %192, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %194 = mhlo.convolution(%193, %13) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x960xf32>, tensor<1x1x960x320xf32>) -> tensor<1x7x7x320xf32>
    %195 = "mhlo.batch_norm_inference"(%194, %12, %12, %12, %12) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>) -> tensor<1x7x7x320xf32>
    %196 = mhlo.convolution(%195, %11) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x320xf32>, tensor<1x1x320x1280xf32>) -> tensor<1x7x7x1280xf32>
    %197 = "mhlo.batch_norm_inference"(%196, %10, %10, %10, %10) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<1x7x7x1280xf32>
    %198 = "mhlo.clamp"(%43, %197, %44) : (tensor<f32>, tensor<1x7x7x1280xf32>, tensor<f32>) -> tensor<1x7x7x1280xf32>
    %199 = "mhlo.reduce"(%198, %43) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %210 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%210) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x1280xf32>, tensor<f32>) -> tensor<1x1280xf32>
    %200 = mhlo.divide %199, %1 : tensor<1x1280xf32>
    %201 = "mhlo.dot"(%200, %9) : (tensor<1x1280xf32>, tensor<1280x1000xf32>) -> tensor<1x1000xf32>
    %202 = mhlo.add %201, %0 : tensor<1x1000xf32>
    %203 = "mhlo.reduce"(%202, %45) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %210 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%210) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %204 = "mhlo.broadcast_in_dim"(%203) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %205 = mhlo.subtract %202, %204 : tensor<1x1000xf32>
    %206 = "mhlo.exponential"(%205) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %207 = "mhlo.reduce"(%206, %43) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %210 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%210) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %208 = "mhlo.broadcast_in_dim"(%207) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %209 = mhlo.divide %206, %208 : tensor<1x1000xf32>
    return %209 : tensor<1x1000xf32>
  }
}

