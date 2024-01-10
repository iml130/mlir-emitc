// RUN: emitc-opt %s --convert-stablehlo-region-ops-to-emitc --convert-stablehlo-to-emitc | emitc-translate --mlir-to-cpp
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 716 : i32}, tf_saved_model.semantics}  {
  func.func @predict(%arg0: tensor<1x224x224x3xf32> {tf._user_specified_name = "args_0", tf_saved_model.index_path = [0]}) -> (tensor<1x1000xf32> {tf_saved_model.index_path = []}) attributes {tf._construction_context = "kEagerRuntime"} {
    %0 = stablehlo.constant dense<5.000000e-01> : tensor<1x1000xf32>
    %1 = stablehlo.constant dense<4.900000e+01> : tensor<1x1280xf32>
    %2 = stablehlo.constant dense<5.000000e-01> : tensor<3x3x1x960xf32>
    %3 = stablehlo.constant dense<5.000000e-01> : tensor<3x3x1x576xf32>
    %4 = stablehlo.constant dense<5.000000e-01> : tensor<3x3x1x384xf32>
    %5 = stablehlo.constant dense<5.000000e-01> : tensor<3x3x1x192xf32>
    %6 = stablehlo.constant dense<5.000000e-01> : tensor<3x3x1x144xf32>
    %7 = stablehlo.constant dense<5.000000e-01> : tensor<3x3x1x96xf32>
    %8 = stablehlo.constant dense<5.000000e-01> : tensor<3x3x1x32xf32>
    %9 = stablehlo.constant dense<5.000000e-01> : tensor<1280x1000xf32>
    %10 = stablehlo.constant dense<5.000000e-01> : tensor<1280xf32>
    %11 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x320x1280xf32>
    %12 = stablehlo.constant dense<5.000000e-01> : tensor<320xf32>
    %13 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x960x320xf32>
    %14 = stablehlo.constant dense<5.000000e-01> : tensor<960xf32>
    %15 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x160x960xf32>
    %16 = stablehlo.constant dense<5.000000e-01> : tensor<160xf32>
    %17 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x960x160xf32>
    %18 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x576x160xf32>
    %19 = stablehlo.constant dense<5.000000e-01> : tensor<576xf32>
    %20 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x96x576xf32>
    %21 = stablehlo.constant dense<5.000000e-01> : tensor<96xf32>
    %22 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x576x96xf32>
    %23 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x384x96xf32>
    %24 = stablehlo.constant dense<5.000000e-01> : tensor<384xf32>
    %25 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x64x384xf32>
    %26 = stablehlo.constant dense<5.000000e-01> : tensor<64xf32>
    %27 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x384x64xf32>
    %28 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x192x64xf32>
    %29 = stablehlo.constant dense<5.000000e-01> : tensor<192xf32>
    %30 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x32x192xf32>
    %31 = stablehlo.constant dense<5.000000e-01> : tensor<32xf32>
    %32 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x192x32xf32>
    %33 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x144x32xf32>
    %34 = stablehlo.constant dense<5.000000e-01> : tensor<144xf32>
    %35 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x24x144xf32>
    %36 = stablehlo.constant dense<5.000000e-01> : tensor<24xf32>
    %37 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x144x24xf32>
    %38 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x96x24xf32>
    %39 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x16x96xf32>
    %40 = stablehlo.constant dense<5.000000e-01> : tensor<16xf32>
    %41 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x32x16xf32>
    %42 = stablehlo.constant dense<5.000000e-01> : tensor<3x3x3x32xf32>
    %43 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %45 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %46 = stablehlo.convolution(%arg0, %42) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x224x224x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32>
    %47 = "stablehlo.batch_norm_inference"(%46, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %48 = "stablehlo.clamp"(%43, %47, %44) : (tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) -> tensor<1x112x112x32xf32>
    %49 = stablehlo.convolution(%48, %8) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<1x112x112x32xf32>, tensor<3x3x1x32xf32>) -> tensor<1x112x112x32xf32>
    %50 = "stablehlo.batch_norm_inference"(%49, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %51 = "stablehlo.clamp"(%43, %50, %44) : (tensor<f32>, tensor<1x112x112x32xf32>, tensor<f32>) -> tensor<1x112x112x32xf32>
    %52 = stablehlo.convolution(%51, %41) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x112x112x32xf32>, tensor<1x1x32x16xf32>) -> tensor<1x112x112x16xf32>
    %53 = "stablehlo.batch_norm_inference"(%52, %40, %40, %40, %40) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %54 = stablehlo.convolution(%53, %39) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x112x112x16xf32>, tensor<1x1x16x96xf32>) -> tensor<1x112x112x96xf32>
    %55 = "stablehlo.batch_norm_inference"(%54, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x112x112x96xf32>
    %56 = "stablehlo.clamp"(%43, %55, %44) : (tensor<f32>, tensor<1x112x112x96xf32>, tensor<f32>) -> tensor<1x112x112x96xf32>
    %57 = "stablehlo.pad"(%56, %43) {edge_padding_high = array<i64: 0, 1, 1, 0>, edge_padding_low = array<i64: 0, 0, 0, 0>, interior_padding = array<i64: 0, 0, 0, 0>} : (tensor<1x112x112x96xf32>, tensor<f32>) -> tensor<1x113x113x96xf32>
    %58 = stablehlo.convolution(%57, %7) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<1x113x113x96xf32>, tensor<3x3x1x96xf32>) -> tensor<1x56x56x96xf32>
    %59 = "stablehlo.batch_norm_inference"(%58, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %60 = "stablehlo.clamp"(%43, %59, %44) : (tensor<f32>, tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x96xf32>
    %61 = stablehlo.convolution(%60, %38) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x96xf32>, tensor<1x1x96x24xf32>) -> tensor<1x56x56x24xf32>
    %62 = "stablehlo.batch_norm_inference"(%61, %36, %36, %36, %36) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %63 = stablehlo.convolution(%62, %35) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) -> tensor<1x56x56x144xf32>
    %64 = "stablehlo.batch_norm_inference"(%63, %34, %34, %34, %34) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %65 = "stablehlo.clamp"(%43, %64, %44) : (tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x56x56x144xf32>
    %66 = stablehlo.convolution(%65, %6) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<1x56x56x144xf32>, tensor<3x3x1x144xf32>) -> tensor<1x56x56x144xf32>
    %67 = "stablehlo.batch_norm_inference"(%66, %34, %34, %34, %34) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %68 = "stablehlo.clamp"(%43, %67, %44) : (tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x56x56x144xf32>
    %69 = stablehlo.convolution(%68, %37) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x144xf32>, tensor<1x1x144x24xf32>) -> tensor<1x56x56x24xf32>
    %70 = "stablehlo.batch_norm_inference"(%69, %36, %36, %36, %36) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x56x56x24xf32>
    %71 = stablehlo.add %62, %70 : tensor<1x56x56x24xf32>
    %72 = stablehlo.convolution(%71, %35) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x24xf32>, tensor<1x1x24x144xf32>) -> tensor<1x56x56x144xf32>
    %73 = "stablehlo.batch_norm_inference"(%72, %34, %34, %34, %34) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x56x56x144xf32>
    %74 = "stablehlo.clamp"(%43, %73, %44) : (tensor<f32>, tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x56x56x144xf32>
    %75 = "stablehlo.pad"(%74, %43) {edge_padding_high = array<i64: 0, 1, 1, 0>, edge_padding_low = array<i64: 0, 0, 0, 0>, interior_padding = array<i64: 0, 0, 0, 0>} : (tensor<1x56x56x144xf32>, tensor<f32>) -> tensor<1x57x57x144xf32>
    %76 = stablehlo.convolution(%75, %6) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<1x57x57x144xf32>, tensor<3x3x1x144xf32>) -> tensor<1x28x28x144xf32>
    %77 = "stablehlo.batch_norm_inference"(%76, %34, %34, %34, %34) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x28x28x144xf32>
    %78 = "stablehlo.clamp"(%43, %77, %44) : (tensor<f32>, tensor<1x28x28x144xf32>, tensor<f32>) -> tensor<1x28x28x144xf32>
    %79 = stablehlo.convolution(%78, %33) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x144xf32>, tensor<1x1x144x32xf32>) -> tensor<1x28x28x32xf32>
    %80 = "stablehlo.batch_norm_inference"(%79, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %81 = stablehlo.convolution(%80, %30) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) -> tensor<1x28x28x192xf32>
    %82 = "stablehlo.batch_norm_inference"(%81, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %83 = "stablehlo.clamp"(%43, %82, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %84 = stablehlo.convolution(%83, %5) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<1x28x28x192xf32>, tensor<3x3x1x192xf32>) -> tensor<1x28x28x192xf32>
    %85 = "stablehlo.batch_norm_inference"(%84, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %86 = "stablehlo.clamp"(%43, %85, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %87 = stablehlo.convolution(%86, %32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) -> tensor<1x28x28x32xf32>
    %88 = "stablehlo.batch_norm_inference"(%87, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %89 = stablehlo.add %80, %88 : tensor<1x28x28x32xf32>
    %90 = stablehlo.convolution(%89, %30) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) -> tensor<1x28x28x192xf32>
    %91 = "stablehlo.batch_norm_inference"(%90, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %92 = "stablehlo.clamp"(%43, %91, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %93 = stablehlo.convolution(%92, %5) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<1x28x28x192xf32>, tensor<3x3x1x192xf32>) -> tensor<1x28x28x192xf32>
    %94 = "stablehlo.batch_norm_inference"(%93, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %95 = "stablehlo.clamp"(%43, %94, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %96 = stablehlo.convolution(%95, %32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x192xf32>, tensor<1x1x192x32xf32>) -> tensor<1x28x28x32xf32>
    %97 = "stablehlo.batch_norm_inference"(%96, %31, %31, %31, %31) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<1x28x28x32xf32>
    %98 = stablehlo.add %89, %97 : tensor<1x28x28x32xf32>
    %99 = stablehlo.convolution(%98, %30) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x28x32xf32>, tensor<1x1x32x192xf32>) -> tensor<1x28x28x192xf32>
    %100 = "stablehlo.batch_norm_inference"(%99, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %101 = "stablehlo.clamp"(%43, %100, %44) : (tensor<f32>, tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x192xf32>
    %102 = "stablehlo.pad"(%101, %43) {edge_padding_high = array<i64: 0, 1, 1, 0>, edge_padding_low = array<i64: 0, 0, 0, 0>, interior_padding = array<i64: 0, 0, 0, 0>} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x29x29x192xf32>
    %103 = stablehlo.convolution(%102, %5) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<1x29x29x192xf32>, tensor<3x3x1x192xf32>) -> tensor<1x14x14x192xf32>
    %104 = "stablehlo.batch_norm_inference"(%103, %29, %29, %29, %29) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<1x14x14x192xf32>
    %105 = "stablehlo.clamp"(%43, %104, %44) : (tensor<f32>, tensor<1x14x14x192xf32>, tensor<f32>) -> tensor<1x14x14x192xf32>
    %106 = stablehlo.convolution(%105, %28) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x192xf32>, tensor<1x1x192x64xf32>) -> tensor<1x14x14x64xf32>
    %107 = "stablehlo.batch_norm_inference"(%106, %26, %26, %26, %26) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %108 = stablehlo.convolution(%107, %25) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %109 = "stablehlo.batch_norm_inference"(%108, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %110 = "stablehlo.clamp"(%43, %109, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %111 = stablehlo.convolution(%110, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %112 = "stablehlo.batch_norm_inference"(%111, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %113 = "stablehlo.clamp"(%43, %112, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %114 = stablehlo.convolution(%113, %27) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) -> tensor<1x14x14x64xf32>
    %115 = "stablehlo.batch_norm_inference"(%114, %26, %26, %26, %26) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %116 = stablehlo.add %107, %115 : tensor<1x14x14x64xf32>
    %117 = stablehlo.convolution(%116, %25) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %118 = "stablehlo.batch_norm_inference"(%117, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %119 = "stablehlo.clamp"(%43, %118, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %120 = stablehlo.convolution(%119, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %121 = "stablehlo.batch_norm_inference"(%120, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %122 = "stablehlo.clamp"(%43, %121, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %123 = stablehlo.convolution(%122, %27) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) -> tensor<1x14x14x64xf32>
    %124 = "stablehlo.batch_norm_inference"(%123, %26, %26, %26, %26) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %125 = stablehlo.add %116, %124 : tensor<1x14x14x64xf32>
    %126 = stablehlo.convolution(%125, %25) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %127 = "stablehlo.batch_norm_inference"(%126, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %128 = "stablehlo.clamp"(%43, %127, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %129 = stablehlo.convolution(%128, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %130 = "stablehlo.batch_norm_inference"(%129, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %131 = "stablehlo.clamp"(%43, %130, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %132 = stablehlo.convolution(%131, %27) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x384xf32>, tensor<1x1x384x64xf32>) -> tensor<1x14x14x64xf32>
    %133 = "stablehlo.batch_norm_inference"(%132, %26, %26, %26, %26) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x14x14x64xf32>
    %134 = stablehlo.add %125, %133 : tensor<1x14x14x64xf32>
    %135 = stablehlo.convolution(%134, %25) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x64xf32>, tensor<1x1x64x384xf32>) -> tensor<1x14x14x384xf32>
    %136 = "stablehlo.batch_norm_inference"(%135, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %137 = "stablehlo.clamp"(%43, %136, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %138 = stablehlo.convolution(%137, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<1x14x14x384xf32>, tensor<3x3x1x384xf32>) -> tensor<1x14x14x384xf32>
    %139 = "stablehlo.batch_norm_inference"(%138, %24, %24, %24, %24) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %140 = "stablehlo.clamp"(%43, %139, %44) : (tensor<f32>, tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x384xf32>
    %141 = stablehlo.convolution(%140, %23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x384xf32>, tensor<1x1x384x96xf32>) -> tensor<1x14x14x96xf32>
    %142 = "stablehlo.batch_norm_inference"(%141, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %143 = stablehlo.convolution(%142, %20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x14x14x576xf32>
    %144 = "stablehlo.batch_norm_inference"(%143, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %145 = "stablehlo.clamp"(%43, %144, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %146 = stablehlo.convolution(%145, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<1x14x14x576xf32>, tensor<3x3x1x576xf32>) -> tensor<1x14x14x576xf32>
    %147 = "stablehlo.batch_norm_inference"(%146, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %148 = "stablehlo.clamp"(%43, %147, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %149 = stablehlo.convolution(%148, %22) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x14x14x96xf32>
    %150 = "stablehlo.batch_norm_inference"(%149, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %151 = stablehlo.add %142, %150 : tensor<1x14x14x96xf32>
    %152 = stablehlo.convolution(%151, %20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x14x14x576xf32>
    %153 = "stablehlo.batch_norm_inference"(%152, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %154 = "stablehlo.clamp"(%43, %153, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %155 = stablehlo.convolution(%154, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<1x14x14x576xf32>, tensor<3x3x1x576xf32>) -> tensor<1x14x14x576xf32>
    %156 = "stablehlo.batch_norm_inference"(%155, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %157 = "stablehlo.clamp"(%43, %156, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %158 = stablehlo.convolution(%157, %22) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x14x14x96xf32>
    %159 = "stablehlo.batch_norm_inference"(%158, %21, %21, %21, %21) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %160 = stablehlo.add %151, %159 : tensor<1x14x14x96xf32>
    %161 = stablehlo.convolution(%160, %20) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x14x14x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x14x14x576xf32>
    %162 = "stablehlo.batch_norm_inference"(%161, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x14x14x576xf32>
    %163 = "stablehlo.clamp"(%43, %162, %44) : (tensor<f32>, tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x14x14x576xf32>
    %164 = "stablehlo.pad"(%163, %43) {edge_padding_high = array<i64: 0, 1, 1, 0>, edge_padding_low = array<i64: 0, 0, 0, 0>, interior_padding = array<i64: 0, 0, 0, 0>} : (tensor<1x14x14x576xf32>, tensor<f32>) -> tensor<1x15x15x576xf32>
    %165 = stablehlo.convolution(%164, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 576 : i64} : (tensor<1x15x15x576xf32>, tensor<3x3x1x576xf32>) -> tensor<1x7x7x576xf32>
    %166 = "stablehlo.batch_norm_inference"(%165, %19, %19, %19, %19) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %167 = "stablehlo.clamp"(%43, %166, %44) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %168 = stablehlo.convolution(%167, %18) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x576xf32>, tensor<1x1x576x160xf32>) -> tensor<1x7x7x160xf32>
    %169 = "stablehlo.batch_norm_inference"(%168, %16, %16, %16, %16) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %170 = stablehlo.convolution(%169, %15) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) -> tensor<1x7x7x960xf32>
    %171 = "stablehlo.batch_norm_inference"(%170, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %172 = "stablehlo.clamp"(%43, %171, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %173 = stablehlo.convolution(%172, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<1x7x7x960xf32>, tensor<3x3x1x960xf32>) -> tensor<1x7x7x960xf32>
    %174 = "stablehlo.batch_norm_inference"(%173, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %175 = "stablehlo.clamp"(%43, %174, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %176 = stablehlo.convolution(%175, %17) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) -> tensor<1x7x7x160xf32>
    %177 = "stablehlo.batch_norm_inference"(%176, %16, %16, %16, %16) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %178 = stablehlo.add %169, %177 : tensor<1x7x7x160xf32>
    %179 = stablehlo.convolution(%178, %15) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) -> tensor<1x7x7x960xf32>
    %180 = "stablehlo.batch_norm_inference"(%179, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %181 = "stablehlo.clamp"(%43, %180, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %182 = stablehlo.convolution(%181, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<1x7x7x960xf32>, tensor<3x3x1x960xf32>) -> tensor<1x7x7x960xf32>
    %183 = "stablehlo.batch_norm_inference"(%182, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %184 = "stablehlo.clamp"(%43, %183, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %185 = stablehlo.convolution(%184, %17) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x960xf32>, tensor<1x1x960x160xf32>) -> tensor<1x7x7x160xf32>
    %186 = "stablehlo.batch_norm_inference"(%185, %16, %16, %16, %16) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<1x7x7x160xf32>
    %187 = stablehlo.add %178, %186 : tensor<1x7x7x160xf32>
    %188 = stablehlo.convolution(%187, %15) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x160xf32>, tensor<1x1x160x960xf32>) -> tensor<1x7x7x960xf32>
    %189 = "stablehlo.batch_norm_inference"(%188, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %190 = "stablehlo.clamp"(%43, %189, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %191 = stablehlo.convolution(%190, %2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<1x7x7x960xf32>, tensor<3x3x1x960xf32>) -> tensor<1x7x7x960xf32>
    %192 = "stablehlo.batch_norm_inference"(%191, %14, %14, %14, %14) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>, tensor<960xf32>) -> tensor<1x7x7x960xf32>
    %193 = "stablehlo.clamp"(%43, %192, %44) : (tensor<f32>, tensor<1x7x7x960xf32>, tensor<f32>) -> tensor<1x7x7x960xf32>
    %194 = stablehlo.convolution(%193, %13) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x960xf32>, tensor<1x1x960x320xf32>) -> tensor<1x7x7x320xf32>
    %195 = "stablehlo.batch_norm_inference"(%194, %12, %12, %12, %12) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>, tensor<320xf32>) -> tensor<1x7x7x320xf32>
    %196 = stablehlo.convolution(%195, %11) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x7x7x320xf32>, tensor<1x1x320x1280xf32>) -> tensor<1x7x7x1280xf32>
    %197 = "stablehlo.batch_norm_inference"(%196, %10, %10, %10, %10) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<1x7x7x1280xf32>
    %198 = "stablehlo.clamp"(%43, %197, %44) : (tensor<f32>, tensor<1x7x7x1280xf32>, tensor<f32>) -> tensor<1x7x7x1280xf32>
    %199 = "stablehlo.reduce"(%198, %43) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %210 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%210) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x1280xf32>, tensor<f32>) -> tensor<1x1280xf32>
    %200 = stablehlo.divide %199, %1 : tensor<1x1280xf32>
    %201 = "stablehlo.dot"(%200, %9) : (tensor<1x1280xf32>, tensor<1280x1000xf32>) -> tensor<1x1000xf32>
    %202 = stablehlo.add %201, %0 : tensor<1x1000xf32>
    %203 = "stablehlo.reduce"(%202, %45) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %210 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%210) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %204 = "stablehlo.broadcast_in_dim"(%203) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %205 = stablehlo.subtract %202, %204 : tensor<1x1000xf32>
    %206 = "stablehlo.exponential"(%205) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %207 = "stablehlo.reduce"(%206, %43) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %210 = stablehlo.add %arg1, %arg2 : tensor<f32>
      "stablehlo.return"(%210) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %208 = "stablehlo.broadcast_in_dim"(%207) {broadcast_dimensions = array<i64: 0>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %209 = stablehlo.divide %206, %208 : tensor<1x1000xf32>
    return %209 : tensor<1x1000xf32>
  }
}

