

"module"() ( {
  "func"() ( {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<i32>):  // no predecessors
    %cst = "std.constant"() {value = dense<-2147483648> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "std.constant"() {value = dense<2147483647> : tensor<i32>} : () -> tensor<i32>
    %cst_1 = "std.constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "mhlo.constant"() {value = dense<true> : tensor<4xi1>} : () -> tensor<4xi1>
    %cst_2 = "std.constant"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %1 = "mhlo.constant"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
    %2 = "mhlo.constant"() {value = dense<2.30258512> : tensor<4xf32>} : () -> tensor<4xf32>
    %3 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<1000x4xf32>} : () -> tensor<1000x4xf32>
    %4 = "mhlo.constant"() {value = dense<false> : tensor<1000x4xi1>} : () -> tensor<1000x4xi1>
    %cst_3 = "std.constant"() {value = dense<"0xE9030000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000010000000100000001000000"> : tensor<1000xi32>} : () -> tensor<1000xi32>
    %cst_4 = "std.constant"() {value = dense<1053357856> : tensor<ui32>} : () -> tensor<ui32>
    %5 = "mhlo.constant"() {value = dense<3468443297> : tensor<ui32>} : () -> tensor<ui32>
    %6 = "mhlo.constant"() {value = dense<814007528> : tensor<ui32>} : () -> tensor<ui32>
    %7 = "mhlo.constant"() {value = dense<2454539055> : tensor<ui32>} : () -> tensor<ui32>
    %8 = "mhlo.constant"() {value = dense<4095070582> : tensor<ui32>} : () -> tensor<ui32>
    %9 = "mhlo.constant"() {value = dense<1440634813> : tensor<ui32>} : () -> tensor<ui32>
    %10 = "mhlo.constant"() {value = dense<3081166340> : tensor<ui32>} : () -> tensor<ui32>
    %11 = "mhlo.constant"() {value = dense<426730571> : tensor<ui32>} : () -> tensor<ui32>
    %12 = "mhlo.constant"() {value = dense<2067262098> : tensor<ui32>} : () -> tensor<ui32>
    %13 = "mhlo.constant"() {value = dense<3707793625> : tensor<ui32>} : () -> tensor<ui32>
    %cst_5 = "std.constant"() {value = dense<38149673> : tensor<ui32>} : () -> tensor<ui32>
    %14 = "mhlo.constant"() {value = dense<2565554390> : tensor<ui32>} : () -> tensor<ui32>
    %15 = "mhlo.constant"() {value = dense<3716387409> : tensor<ui32>} : () -> tensor<ui32>
    %16 = "mhlo.constant"() {value = dense<572253132> : tensor<ui32>} : () -> tensor<ui32>
    %17 = "mhlo.constant"() {value = dense<1723086151> : tensor<ui32>} : () -> tensor<ui32>
    %18 = "mhlo.constant"() {value = dense<2873919170> : tensor<ui32>} : () -> tensor<ui32>
    %19 = "mhlo.constant"() {value = dense<4024752189> : tensor<ui32>} : () -> tensor<ui32>
    %20 = "mhlo.constant"() {value = dense<880617912> : tensor<ui32>} : () -> tensor<ui32>
    %21 = "mhlo.constant"() {value = dense<2031450931> : tensor<ui32>} : () -> tensor<ui32>
    %22 = "mhlo.constant"() {value = dense<3182283950> : tensor<ui32>} : () -> tensor<ui32>
    %cst_6 = "std.constant"() {value = dense<3449720151> : tensor<ui64>} : () -> tensor<ui64>
    %23 = "mhlo.constant"() {value = dense<0> : tensor<ui64>} : () -> tensor<ui64>
    %cst_7 = "std.constant"() {value = dense<3528531795> : tensor<ui64>} : () -> tensor<ui64>
    %cst_8 = "std.constant"() {value = dense<32> : tensor<ui64>} : () -> tensor<ui64>
    %cst_9 = "std.constant"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %24 = "mhlo.constant"() {value = dense<6.28318548> : tensor<2xf32>} : () -> tensor<2xf32>
    %25 = "mhlo.constant"() {value = dense<-2.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %26 = "mhlo.constant"() {value = dense<1.000000e-07> : tensor<2xf32>} : () -> tensor<2xf32>
    %27 = "mhlo.constant"() {value = dense<-5.000000e-01> : tensor<4xf32>} : () -> tensor<4xf32>
    %cst_10 = "std.constant"() {value = dense<0.918938517> : tensor<f32>} : () -> tensor<f32>
    %28 = "mhlo.constant"() {value = dense<-1.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
    %29 = "mhlo.constant"() {value = dense<9> : tensor<4xui32>} : () -> tensor<4xui32>
    %30 = "mhlo.constant"() {value = dense<1.1920929E-7> : tensor<4xf32>} : () -> tensor<4xf32>
    %31 = "mhlo.constant"() {value = dense<1.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
    %32 = "mhlo.constant"() {value = dense<5.000000e-01> : tensor<4xf32>} : () -> tensor<4xf32>
    %33 = "mhlo.constant"() {value = dense<2.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
    %34 = "mhlo.constant"() {value = dense<0xFF800000> : tensor<4xf32>} : () -> tensor<4xf32>
    %35 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
    %cst_11 = "std.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_12 = "std.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst_13 = "std.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_14 = "std.constant"() {value = dense<1000> : tensor<i32>} : () -> tensor<i32>
    %36 = "mhlo.log"(%arg0) {name = "log.19"} : (tensor<4xf32>) -> tensor<4xf32>
    %37 = "mhlo.exponential"(%36) {name = "exponential.24"} : (tensor<4xf32>) -> tensor<4xf32>
    %38 = "mhlo.log"(%37) {name = "log.25"} : (tensor<4xf32>) -> tensor<4xf32>
    %39 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.240"} : (tensor<f32>) -> tensor<4xf32>
    %40 = "mhlo.divide"(%38, %39) {name = "divide.241"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %41 = "mhlo.divide"(%arg2, %arg1) {name = "divide.253"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %42 = "mhlo.broadcast_in_dim"(%41) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.254"} : (tensor<f32>) -> tensor<4xf32>
    %43 = "mhlo.subtract"(%40, %42) {name = "subtract.255"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %44 = "mhlo.multiply"(%43, %28) {name = "multiply.258"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %45 = "mhlo.rng_uniform"(%cst, %cst_0, %cst_1) : (tensor<i32>, tensor<i32>, tensor<1xi64>) -> tensor<2xi32>
    %46 = "mhlo.exponential"(%36) {name = "exponential.20"} : (tensor<4xf32>) -> tensor<4xf32>
    %47 = "mhlo.log"(%46) {name = "log.21"} : (tensor<4xf32>) -> tensor<4xf32>
    %48 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.231"} : (tensor<f32>) -> tensor<4xf32>
    %49 = "mhlo.divide"(%47, %48) {name = "divide.232"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %50 = "mhlo.divide"(%arg2, %arg1) {name = "divide.242"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %51 = "mhlo.broadcast_in_dim"(%50) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.243"} : (tensor<f32>) -> tensor<4xf32>
    %52 = "mhlo.subtract"(%49, %51) {name = "subtract.244"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %53 = "mhlo.multiply"(%52, %52) {name = "multiply.245"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %54 = "mhlo.multiply"(%53, %27) {name = "multiply.248"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %55 = "mhlo.log"(%arg1) {name = "log.228"} : (tensor<f32>) -> tensor<f32>
    %56 = "mhlo.add"(%55, %cst_10) {name = "add.230"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %57 = "mhlo.broadcast_in_dim"(%56) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.249"} : (tensor<f32>) -> tensor<4xf32>
    %58 = "mhlo.subtract"(%54, %57) {name = "subtract.250"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %59 = "mhlo.log"(%46) {name = "log.22"} : (tensor<4xf32>) -> tensor<4xf32>
    %60 = "mhlo.multiply"(%59, %28) {name = "multiply.66"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %61 = "mhlo.multiply"(%60, %31) {name = "multiply.69"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %62 = "mhlo.add"(%58, %61) {name = "add.251"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %63 = "mhlo.negate"(%36) {name = "negate.23"} : (tensor<4xf32>) -> tensor<4xf32>
    %64 = "mhlo.multiply"(%63, %31) {name = "multiply.59"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %65 = "mhlo.negate"(%64) {name = "negate.60"} : (tensor<4xf32>) -> tensor<4xf32>
    %66 = "mhlo.add"(%65, %35) {name = "add.63"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %67 = "mhlo.add"(%62, %66) {name = "add.252"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %68 = "mhlo.divide"(%31, %37) {name = "divide.161"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %69 = "mhlo.multiply"(%68, %28) {name = "multiply.164"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %70 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.270"} : (tensor<f32>) -> tensor<4xf32>
    %71 = "mhlo.divide"(%44, %70) {name = "divide.271"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %72 = "mhlo.divide"(%31, %37) {name = "divide.275"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %73 = "mhlo.multiply"(%71, %72) {name = "multiply.276"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %74 = "mhlo.add"(%69, %73) {name = "add.277"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %75 = "mhlo.multiply"(%74, %37) {name = "multiply.278"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %76 = "mhlo.add"(%75, %31) {name = "add.281"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %77 = "mhlo.log"(%arg3) {name = "log.316"} : (tensor<4xf32>) -> tensor<4xf32>
    %78 = "mhlo.add"(%77, %2) {name = "add.319"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %79 = "mhlo.tuple"(%3, %cst_14) {name = "tuple.217"} : (tensor<1000x4xf32>, tensor<i32>) -> tuple<tensor<1000x4xf32>, tensor<i32>>
    %80 = "mhlo.tuple"(%4, %cst_14) {name = "tuple.227"} : (tensor<1000x4xi1>, tensor<i32>) -> tuple<tensor<1000x4xi1>, tensor<i32>>
    %81 = "mhlo.tuple"(%cst_3, %cst_14) {name = "tuple.198"} : (tensor<1000xi32>, tensor<i32>) -> tuple<tensor<1000xi32>, tensor<i32>>
    %82 = "mhlo.tuple"(%cst_13, %cst_9, %cst_13, %45, %arg0, %36, %35, %67, %76, %35, %35, %arg3, %cst_2, %0, %35, %36, %35, %67, %76, %35, %35, %arg3, %cst_2, %1, %1, %arg4, %78, %arg5, %arg6, %arg7, %35, %35, %cst_13, %arg3, %cst_13, %79, %80, %81, %arg1, %arg2, %arg8) {name = "tuple.349"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>
    %83 = "mhlo.while"(%82) ( {
    ^bb0(%arg9: tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>):  // no predecessors
      %89 = "mhlo.get_tuple_element"(%arg9) {index = 2 : i32, name = "get-tuple-element.2495"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %90 = "mhlo.compare"(%89, %cst_14) {comparison_direction = "LT", name = "compare.2535"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "mhlo.return"(%90) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg9: tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>):  // no predecessors
      %89 = "mhlo.get_tuple_element"(%arg9) {index = 3 : i32, name = "get-tuple-element.2345"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<2xi32>
      %90 = "mhlo.get_tuple_element"(%arg9) {index = 4 : i32, name = "get-tuple-element.2346"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %91 = "mhlo.get_tuple_element"(%arg9) {index = 5 : i32, name = "get-tuple-element.2347"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %92 = "mhlo.get_tuple_element"(%arg9) {index = 6 : i32, name = "get-tuple-element.2348"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %93 = "mhlo.get_tuple_element"(%arg9) {index = 7 : i32, name = "get-tuple-element.2349"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %94 = "mhlo.get_tuple_element"(%arg9) {index = 8 : i32, name = "get-tuple-element.2350"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %95 = "mhlo.get_tuple_element"(%arg9) {index = 9 : i32, name = "get-tuple-element.2351"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %96 = "mhlo.get_tuple_element"(%arg9) {index = 10 : i32, name = "get-tuple-element.2352"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %97 = "mhlo.get_tuple_element"(%arg9) {index = 11 : i32, name = "get-tuple-element.2353"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %98 = "mhlo.get_tuple_element"(%arg9) {index = 12 : i32, name = "get-tuple-element.2354"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %99 = "mhlo.get_tuple_element"(%arg9) {index = 13 : i32, name = "get-tuple-element.2355"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xi1>
      %100 = "mhlo.get_tuple_element"(%arg9) {index = 14 : i32, name = "get-tuple-element.2356"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %101 = "mhlo.get_tuple_element"(%arg9) {index = 15 : i32, name = "get-tuple-element.2357"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %102 = "mhlo.get_tuple_element"(%arg9) {index = 16 : i32, name = "get-tuple-element.2358"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %103 = "mhlo.get_tuple_element"(%arg9) {index = 17 : i32, name = "get-tuple-element.2359"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %104 = "mhlo.get_tuple_element"(%arg9) {index = 18 : i32, name = "get-tuple-element.2360"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %105 = "mhlo.get_tuple_element"(%arg9) {index = 19 : i32, name = "get-tuple-element.2361"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %106 = "mhlo.get_tuple_element"(%arg9) {index = 20 : i32, name = "get-tuple-element.2362"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %107 = "mhlo.get_tuple_element"(%arg9) {index = 21 : i32, name = "get-tuple-element.2363"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %108 = "mhlo.get_tuple_element"(%arg9) {index = 22 : i32, name = "get-tuple-element.2364"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %109 = "mhlo.get_tuple_element"(%arg9) {index = 23 : i32, name = "get-tuple-element.2365"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<2xi32>
      %110 = "mhlo.get_tuple_element"(%arg9) {index = 24 : i32, name = "get-tuple-element.2366"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<2xi32>
      %111 = "mhlo.get_tuple_element"(%arg9) {index = 25 : i32, name = "get-tuple-element.2367"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %112 = "mhlo.get_tuple_element"(%arg9) {index = 26 : i32, name = "get-tuple-element.2368"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %113 = "mhlo.get_tuple_element"(%arg9) {index = 27 : i32, name = "get-tuple-element.2369"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %114 = "mhlo.get_tuple_element"(%arg9) {index = 28 : i32, name = "get-tuple-element.2370"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %115 = "mhlo.get_tuple_element"(%arg9) {index = 29 : i32, name = "get-tuple-element.2371"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %116 = "mhlo.get_tuple_element"(%arg9) {index = 30 : i32, name = "get-tuple-element.2372"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %117 = "mhlo.get_tuple_element"(%arg9) {index = 31 : i32, name = "get-tuple-element.2373"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %118 = "mhlo.get_tuple_element"(%arg9) {index = 32 : i32, name = "get-tuple-element.2374"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %119 = "mhlo.get_tuple_element"(%arg9) {index = 33 : i32, name = "get-tuple-element.2375"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %120 = "mhlo.get_tuple_element"(%arg9) {index = 37 : i32, name = "get-tuple-element.2379"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tuple<tensor<1000xi32>, tensor<i32>>
      %121 = "mhlo.get_tuple_element"(%120) {index = 0 : i32, name = "get-tuple-element.2390"} : (tuple<tensor<1000xi32>, tensor<i32>>) -> tensor<1000xi32>
      %122 = "mhlo.get_tuple_element"(%arg9) {index = 2 : i32, name = "get-tuple-element.2344"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %123 = "mhlo.dynamic-slice"(%121, %122) {slice_sizes = dense<1> : tensor<1xi64>} : (tensor<1000xi32>, tensor<i32>) -> tensor<1xi32>
      %124 = "mhlo.reshape"(%123) {name = "reshape.2392"} : (tensor<1xi32>) -> tensor<i32>
      %125 = "mhlo.get_tuple_element"(%arg9) {index = 38 : i32, name = "get-tuple-element.2380"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %126 = "mhlo.get_tuple_element"(%arg9) {index = 39 : i32, name = "get-tuple-element.2381"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %127 = "mhlo.get_tuple_element"(%arg9) {index = 40 : i32, name = "get-tuple-element.2382"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %128 = "mhlo.tuple"(%cst_13, %cst_9, %cst_13, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %124, %125, %126, %127) {name = "tuple.2399"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>
      %129 = "mhlo.while"(%128) ( {
      ^bb0(%arg10: tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>):  // no predecessors
        %180 = "mhlo.get_tuple_element"(%arg10) {index = 2 : i32, name = "get-tuple-element.2298"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
        %181 = "mhlo.get_tuple_element"(%arg10) {index = 34 : i32, name = "get-tuple-element.2330"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
        %182 = "mhlo.compare"(%180, %181) {comparison_direction = "LT", name = "compare.2334"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "mhlo.return"(%182) : (tensor<i1>) -> ()
      },  {
      ^bb0(%arg10: tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>):  // no predecessors
        %180 = "mhlo.get_tuple_element"(%arg10) {index = 3 : i32, name = "get-tuple-element.620"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<2xi32>
        %181 = "mhlo.slice"(%180) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %182 = "mhlo.reshape"(%181) {name = "reshape.726"} : (tensor<1xi32>) -> tensor<i32>
        %183 = "mhlo.convert"(%182) {name = "convert.729"} : (tensor<i32>) -> tensor<ui64>
        %184 = "mhlo.slice"(%180) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %185 = "mhlo.reshape"(%184) {name = "reshape.728"} : (tensor<1xi32>) -> tensor<i32>
        %186 = "mhlo.convert"(%185) {name = "convert.730"} : (tensor<i32>) -> tensor<ui64>
        %187 = "mhlo.shift_left"(%186, %cst_8) {name = "shift-left.732"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %188 = "mhlo.or"(%183, %187) {name = "or.733"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %189 = "mhlo.convert"(%188) {name = "convert.736"} : (tensor<ui64>) -> tensor<ui32>
        %190 = "mhlo.convert"(%189) {name = "convert.739"} : (tensor<ui32>) -> tensor<ui64>
        %191 = "mhlo.convert"(%190) {name = "convert.741"} : (tensor<ui64>) -> tensor<ui32>
        %192 = "mhlo.convert"(%191) {name = "convert.751"} : (tensor<ui32>) -> tensor<ui64>
        %193 = "mhlo.multiply"(%192, %cst_7) {name = "multiply.753"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %194 = "mhlo.shift_right_logical"(%193, %cst_8) {name = "shift-right-logical.756"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %195 = "mhlo.convert"(%194) {name = "convert.757"} : (tensor<ui64>) -> tensor<ui32>
        %196 = "mhlo.shift_right_logical"(%188, %cst_8) {name = "shift-right-logical.737"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %197 = "mhlo.convert"(%196) {name = "convert.738"} : (tensor<ui64>) -> tensor<ui32>
        %198 = "mhlo.convert"(%197) {name = "convert.740"} : (tensor<ui32>) -> tensor<ui64>
        %199 = "mhlo.shift_right_logical"(%198, %cst_8) {name = "shift-right-logical.747"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %200 = "mhlo.convert"(%199) {name = "convert.748"} : (tensor<ui64>) -> tensor<ui32>
        %201 = "mhlo.xor"(%195, %200) {name = "xor.767"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %202 = "mhlo.xor"(%201, %cst_5) {name = "xor.768"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %203 = "mhlo.convert"(%202) {name = "convert.780"} : (tensor<ui32>) -> tensor<ui64>
        %204 = "mhlo.multiply"(%203, %cst_6) {name = "multiply.782"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %205 = "mhlo.shift_right_logical"(%204, %cst_8) {name = "shift-right-logical.785"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %206 = "mhlo.convert"(%205) {name = "convert.786"} : (tensor<ui64>) -> tensor<ui32>
        %207 = "mhlo.convert"(%198) {name = "convert.745"} : (tensor<ui64>) -> tensor<ui32>
        %208 = "mhlo.convert"(%207) {name = "convert.758"} : (tensor<ui32>) -> tensor<ui64>
        %209 = "mhlo.multiply"(%208, %cst_6) {name = "multiply.760"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %210 = "mhlo.convert"(%209) {name = "convert.761"} : (tensor<ui64>) -> tensor<ui32>
        %211 = "mhlo.xor"(%206, %210) {name = "xor.787"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %212 = "mhlo.xor"(%211, %13) {name = "xor.788"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %213 = "mhlo.convert"(%212) {name = "convert.795"} : (tensor<ui32>) -> tensor<ui64>
        %214 = "mhlo.multiply"(%213, %cst_7) {name = "multiply.797"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %215 = "mhlo.shift_right_logical"(%214, %cst_8) {name = "shift-right-logical.800"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %216 = "mhlo.convert"(%215) {name = "convert.801"} : (tensor<ui64>) -> tensor<ui32>
        %217 = "mhlo.shift_right_logical"(%209, %cst_8) {name = "shift-right-logical.763"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %218 = "mhlo.convert"(%217) {name = "convert.764"} : (tensor<ui64>) -> tensor<ui32>
        %219 = "mhlo.shift_right_logical"(%190, %cst_8) {name = "shift-right-logical.743"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %220 = "mhlo.convert"(%219) {name = "convert.744"} : (tensor<ui64>) -> tensor<ui32>
        %221 = "mhlo.xor"(%218, %220) {name = "xor.765"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %222 = "mhlo.xor"(%221, %cst_4) {name = "xor.766"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %223 = "mhlo.convert"(%222) {name = "convert.773"} : (tensor<ui32>) -> tensor<ui64>
        %224 = "mhlo.multiply"(%223, %cst_7) {name = "multiply.775"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %225 = "mhlo.convert"(%224) {name = "convert.776"} : (tensor<ui64>) -> tensor<ui32>
        %226 = "mhlo.xor"(%216, %225) {name = "xor.811"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %227 = "mhlo.xor"(%226, %21) {name = "xor.812"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %228 = "mhlo.convert"(%227) {name = "convert.824"} : (tensor<ui32>) -> tensor<ui64>
        %229 = "mhlo.multiply"(%228, %cst_6) {name = "multiply.826"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %230 = "mhlo.shift_right_logical"(%229, %cst_8) {name = "shift-right-logical.829"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %231 = "mhlo.convert"(%230) {name = "convert.830"} : (tensor<ui64>) -> tensor<ui32>
        %232 = "mhlo.shift_right_logical"(%224, %cst_8) {name = "shift-right-logical.778"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %233 = "mhlo.convert"(%232) {name = "convert.779"} : (tensor<ui64>) -> tensor<ui32>
        %234 = "mhlo.convert"(%193) {name = "convert.754"} : (tensor<ui64>) -> tensor<ui32>
        %235 = "mhlo.xor"(%233, %234) {name = "xor.789"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %236 = "mhlo.xor"(%235, %22) {name = "xor.790"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %237 = "mhlo.convert"(%236) {name = "convert.802"} : (tensor<ui32>) -> tensor<ui64>
        %238 = "mhlo.multiply"(%237, %cst_6) {name = "multiply.804"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %239 = "mhlo.convert"(%238) {name = "convert.805"} : (tensor<ui64>) -> tensor<ui32>
        %240 = "mhlo.xor"(%231, %239) {name = "xor.831"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %241 = "mhlo.xor"(%240, %11) {name = "xor.832"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %242 = "mhlo.convert"(%241) {name = "convert.839"} : (tensor<ui32>) -> tensor<ui64>
        %243 = "mhlo.multiply"(%242, %cst_7) {name = "multiply.841"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %244 = "mhlo.shift_right_logical"(%243, %cst_8) {name = "shift-right-logical.844"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %245 = "mhlo.convert"(%244) {name = "convert.845"} : (tensor<ui64>) -> tensor<ui32>
        %246 = "mhlo.shift_right_logical"(%238, %cst_8) {name = "shift-right-logical.807"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %247 = "mhlo.convert"(%246) {name = "convert.808"} : (tensor<ui64>) -> tensor<ui32>
        %248 = "mhlo.convert"(%204) {name = "convert.783"} : (tensor<ui64>) -> tensor<ui32>
        %249 = "mhlo.xor"(%247, %248) {name = "xor.809"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %250 = "mhlo.xor"(%249, %12) {name = "xor.810"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %251 = "mhlo.convert"(%250) {name = "convert.817"} : (tensor<ui32>) -> tensor<ui64>
        %252 = "mhlo.multiply"(%251, %cst_7) {name = "multiply.819"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %253 = "mhlo.convert"(%252) {name = "convert.820"} : (tensor<ui64>) -> tensor<ui32>
        %254 = "mhlo.xor"(%245, %253) {name = "xor.855"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %255 = "mhlo.xor"(%254, %19) {name = "xor.856"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %256 = "mhlo.convert"(%255) {name = "convert.868"} : (tensor<ui32>) -> tensor<ui64>
        %257 = "mhlo.multiply"(%256, %cst_6) {name = "multiply.870"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %258 = "mhlo.shift_right_logical"(%257, %cst_8) {name = "shift-right-logical.873"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %259 = "mhlo.convert"(%258) {name = "convert.874"} : (tensor<ui64>) -> tensor<ui32>
        %260 = "mhlo.shift_right_logical"(%252, %cst_8) {name = "shift-right-logical.822"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %261 = "mhlo.convert"(%260) {name = "convert.823"} : (tensor<ui64>) -> tensor<ui32>
        %262 = "mhlo.convert"(%214) {name = "convert.798"} : (tensor<ui64>) -> tensor<ui32>
        %263 = "mhlo.xor"(%261, %262) {name = "xor.833"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %264 = "mhlo.xor"(%263, %20) {name = "xor.834"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %265 = "mhlo.convert"(%264) {name = "convert.846"} : (tensor<ui32>) -> tensor<ui64>
        %266 = "mhlo.multiply"(%265, %cst_6) {name = "multiply.848"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %267 = "mhlo.convert"(%266) {name = "convert.849"} : (tensor<ui64>) -> tensor<ui32>
        %268 = "mhlo.xor"(%259, %267) {name = "xor.875"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %269 = "mhlo.xor"(%268, %9) {name = "xor.876"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %270 = "mhlo.convert"(%269) {name = "convert.883"} : (tensor<ui32>) -> tensor<ui64>
        %271 = "mhlo.multiply"(%270, %cst_7) {name = "multiply.885"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %272 = "mhlo.shift_right_logical"(%271, %cst_8) {name = "shift-right-logical.888"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %273 = "mhlo.convert"(%272) {name = "convert.889"} : (tensor<ui64>) -> tensor<ui32>
        %274 = "mhlo.shift_right_logical"(%266, %cst_8) {name = "shift-right-logical.851"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %275 = "mhlo.convert"(%274) {name = "convert.852"} : (tensor<ui64>) -> tensor<ui32>
        %276 = "mhlo.convert"(%229) {name = "convert.827"} : (tensor<ui64>) -> tensor<ui32>
        %277 = "mhlo.xor"(%275, %276) {name = "xor.853"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %278 = "mhlo.xor"(%277, %10) {name = "xor.854"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %279 = "mhlo.convert"(%278) {name = "convert.861"} : (tensor<ui32>) -> tensor<ui64>
        %280 = "mhlo.multiply"(%279, %cst_7) {name = "multiply.863"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %281 = "mhlo.convert"(%280) {name = "convert.864"} : (tensor<ui64>) -> tensor<ui32>
        %282 = "mhlo.xor"(%273, %281) {name = "xor.899"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %283 = "mhlo.xor"(%282, %17) {name = "xor.900"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %284 = "mhlo.convert"(%283) {name = "convert.912"} : (tensor<ui32>) -> tensor<ui64>
        %285 = "mhlo.multiply"(%284, %cst_6) {name = "multiply.914"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %286 = "mhlo.shift_right_logical"(%285, %cst_8) {name = "shift-right-logical.917"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %287 = "mhlo.convert"(%286) {name = "convert.918"} : (tensor<ui64>) -> tensor<ui32>
        %288 = "mhlo.shift_right_logical"(%280, %cst_8) {name = "shift-right-logical.866"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %289 = "mhlo.convert"(%288) {name = "convert.867"} : (tensor<ui64>) -> tensor<ui32>
        %290 = "mhlo.convert"(%243) {name = "convert.842"} : (tensor<ui64>) -> tensor<ui32>
        %291 = "mhlo.xor"(%289, %290) {name = "xor.877"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %292 = "mhlo.xor"(%291, %18) {name = "xor.878"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %293 = "mhlo.convert"(%292) {name = "convert.890"} : (tensor<ui32>) -> tensor<ui64>
        %294 = "mhlo.multiply"(%293, %cst_6) {name = "multiply.892"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %295 = "mhlo.convert"(%294) {name = "convert.893"} : (tensor<ui64>) -> tensor<ui32>
        %296 = "mhlo.xor"(%287, %295) {name = "xor.919"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %297 = "mhlo.xor"(%296, %7) {name = "xor.920"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %298 = "mhlo.convert"(%297) {name = "convert.927"} : (tensor<ui32>) -> tensor<ui64>
        %299 = "mhlo.multiply"(%298, %cst_7) {name = "multiply.929"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %300 = "mhlo.shift_right_logical"(%299, %cst_8) {name = "shift-right-logical.932"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %301 = "mhlo.convert"(%300) {name = "convert.933"} : (tensor<ui64>) -> tensor<ui32>
        %302 = "mhlo.shift_right_logical"(%294, %cst_8) {name = "shift-right-logical.895"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %303 = "mhlo.convert"(%302) {name = "convert.896"} : (tensor<ui64>) -> tensor<ui32>
        %304 = "mhlo.convert"(%257) {name = "convert.871"} : (tensor<ui64>) -> tensor<ui32>
        %305 = "mhlo.xor"(%303, %304) {name = "xor.897"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %306 = "mhlo.xor"(%305, %8) {name = "xor.898"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %307 = "mhlo.convert"(%306) {name = "convert.905"} : (tensor<ui32>) -> tensor<ui64>
        %308 = "mhlo.multiply"(%307, %cst_7) {name = "multiply.907"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %309 = "mhlo.convert"(%308) {name = "convert.908"} : (tensor<ui64>) -> tensor<ui32>
        %310 = "mhlo.xor"(%301, %309) {name = "xor.943"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %311 = "mhlo.xor"(%310, %15) {name = "xor.944"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %312 = "mhlo.convert"(%311) {name = "convert.956"} : (tensor<ui32>) -> tensor<ui64>
        %313 = "mhlo.multiply"(%312, %cst_6) {name = "multiply.958"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %314 = "mhlo.shift_right_logical"(%313, %cst_8) {name = "shift-right-logical.961"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %315 = "mhlo.convert"(%314) {name = "convert.962"} : (tensor<ui64>) -> tensor<ui32>
        %316 = "mhlo.shift_right_logical"(%308, %cst_8) {name = "shift-right-logical.910"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %317 = "mhlo.convert"(%316) {name = "convert.911"} : (tensor<ui64>) -> tensor<ui32>
        %318 = "mhlo.convert"(%271) {name = "convert.886"} : (tensor<ui64>) -> tensor<ui32>
        %319 = "mhlo.xor"(%317, %318) {name = "xor.921"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %320 = "mhlo.xor"(%319, %16) {name = "xor.922"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %321 = "mhlo.convert"(%320) {name = "convert.934"} : (tensor<ui32>) -> tensor<ui64>
        %322 = "mhlo.multiply"(%321, %cst_6) {name = "multiply.936"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %323 = "mhlo.convert"(%322) {name = "convert.937"} : (tensor<ui64>) -> tensor<ui32>
        %324 = "mhlo.xor"(%315, %323) {name = "xor.963"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %325 = "mhlo.xor"(%324, %5) {name = "xor.964"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %326 = "mhlo.convert"(%325) {name = "convert.985"} : (tensor<ui32>) -> tensor<ui64>
        %327 = "mhlo.convert"(%313) {name = "convert.959"} : (tensor<ui64>) -> tensor<ui32>
        %328 = "mhlo.convert"(%327) {name = "convert.986"} : (tensor<ui32>) -> tensor<ui64>
        %329 = "mhlo.shift_left"(%328, %cst_8) {name = "shift-left.988"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %330 = "mhlo.or"(%326, %329) {name = "or.989"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %331 = "mhlo.reshape"(%330) {name = "reshape.990"} : (tensor<ui64>) -> tensor<1xui64>
        %332 = "mhlo.shift_left"(%23, %cst_8) {name = "shift-left.975"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %333 = "mhlo.or"(%332, %23) {name = "or.976"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %334 = "mhlo.reshape"(%333) {name = "reshape.982"} : (tensor<ui64>) -> tensor<1xui64>
        %335 = "mhlo.shift_right_logical"(%322, %cst_8) {name = "shift-right-logical.939"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %336 = "mhlo.convert"(%335) {name = "convert.940"} : (tensor<ui64>) -> tensor<ui32>
        %337 = "mhlo.convert"(%285) {name = "convert.915"} : (tensor<ui64>) -> tensor<ui32>
        %338 = "mhlo.xor"(%336, %337) {name = "xor.941"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %339 = "mhlo.xor"(%338, %6) {name = "xor.942"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %340 = "mhlo.convert"(%339) {name = "convert.949"} : (tensor<ui32>) -> tensor<ui64>
        %341 = "mhlo.multiply"(%340, %cst_7) {name = "multiply.951"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %342 = "mhlo.shift_right_logical"(%341, %cst_8) {name = "shift-right-logical.954"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %343 = "mhlo.convert"(%342) {name = "convert.955"} : (tensor<ui64>) -> tensor<ui32>
        %344 = "mhlo.convert"(%299) {name = "convert.930"} : (tensor<ui64>) -> tensor<ui32>
        %345 = "mhlo.xor"(%343, %344) {name = "xor.965"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %346 = "mhlo.xor"(%345, %14) {name = "xor.966"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %347 = "mhlo.convert"(%346) {name = "convert.977"} : (tensor<ui32>) -> tensor<ui64>
        %348 = "mhlo.convert"(%341) {name = "convert.952"} : (tensor<ui64>) -> tensor<ui32>
        %349 = "mhlo.convert"(%348) {name = "convert.978"} : (tensor<ui32>) -> tensor<ui64>
        %350 = "mhlo.shift_left"(%349, %cst_8) {name = "shift-left.980"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %351 = "mhlo.or"(%347, %350) {name = "or.981"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %352 = "mhlo.reshape"(%351) {name = "reshape.983"} : (tensor<ui64>) -> tensor<1xui64>
        %353 = "mhlo.concatenate"(%334, %352) {dimension = 0 : i64} : (tensor<1xui64>, tensor<1xui64>) -> tensor<2xui64>
        %354 = "mhlo.concatenate"(%331, %353) {dimension = 0 : i64} : (tensor<1xui64>, tensor<2xui64>) -> tensor<3xui64>
        %355 = "mhlo.rng_bit_generator"(%354) {rng_algorithm = 2 : i32} : (tensor<3xui64>) -> tuple<tensor<3xui64>, tensor<2x2xui32>>
        %356 = "mhlo.get_tuple_element"(%355) {index = 1 : i32, name = "get-tuple-element.993"} : (tuple<tensor<3xui64>, tensor<2x2xui32>>) -> tensor<2x2xui32>
        %357 = "mhlo.bitcast_convert"(%356) {name = "bitcast-convert.995"} : (tensor<2x2xui32>) -> tensor<2x2xi32>
        %358 = "mhlo.slice"(%357) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xi32>) -> tensor<1x2xi32>
        %359 = "mhlo.reshape"(%358) {name = "reshape.997"} : (tensor<1x2xi32>) -> tensor<2xi32>
        %360 = "mhlo.slice"(%359) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %361 = "mhlo.reshape"(%360) {name = "reshape.1001"} : (tensor<1xi32>) -> tensor<i32>
        %362 = "mhlo.convert"(%361) {name = "convert.1004"} : (tensor<i32>) -> tensor<ui64>
        %363 = "mhlo.slice"(%359) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %364 = "mhlo.reshape"(%363) {name = "reshape.1003"} : (tensor<1xi32>) -> tensor<i32>
        %365 = "mhlo.convert"(%364) {name = "convert.1005"} : (tensor<i32>) -> tensor<ui64>
        %366 = "mhlo.shift_left"(%365, %cst_8) {name = "shift-left.1007"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %367 = "mhlo.or"(%362, %366) {name = "or.1008"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %368 = "mhlo.convert"(%367) {name = "convert.1011"} : (tensor<ui64>) -> tensor<ui32>
        %369 = "mhlo.convert"(%368) {name = "convert.1014"} : (tensor<ui32>) -> tensor<ui64>
        %370 = "mhlo.convert"(%369) {name = "convert.1016"} : (tensor<ui64>) -> tensor<ui32>
        %371 = "mhlo.convert"(%370) {name = "convert.1026"} : (tensor<ui32>) -> tensor<ui64>
        %372 = "mhlo.multiply"(%371, %cst_7) {name = "multiply.1028"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %373 = "mhlo.shift_right_logical"(%372, %cst_8) {name = "shift-right-logical.1031"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %374 = "mhlo.convert"(%373) {name = "convert.1032"} : (tensor<ui64>) -> tensor<ui32>
        %375 = "mhlo.shift_right_logical"(%367, %cst_8) {name = "shift-right-logical.1012"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %376 = "mhlo.convert"(%375) {name = "convert.1013"} : (tensor<ui64>) -> tensor<ui32>
        %377 = "mhlo.convert"(%376) {name = "convert.1015"} : (tensor<ui32>) -> tensor<ui64>
        %378 = "mhlo.shift_right_logical"(%377, %cst_8) {name = "shift-right-logical.1022"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %379 = "mhlo.convert"(%378) {name = "convert.1023"} : (tensor<ui64>) -> tensor<ui32>
        %380 = "mhlo.xor"(%374, %379) {name = "xor.1042"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %381 = "mhlo.xor"(%380, %cst_5) {name = "xor.1043"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %382 = "mhlo.convert"(%381) {name = "convert.1055"} : (tensor<ui32>) -> tensor<ui64>
        %383 = "mhlo.multiply"(%382, %cst_6) {name = "multiply.1057"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %384 = "mhlo.shift_right_logical"(%383, %cst_8) {name = "shift-right-logical.1060"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %385 = "mhlo.convert"(%384) {name = "convert.1061"} : (tensor<ui64>) -> tensor<ui32>
        %386 = "mhlo.convert"(%377) {name = "convert.1020"} : (tensor<ui64>) -> tensor<ui32>
        %387 = "mhlo.convert"(%386) {name = "convert.1033"} : (tensor<ui32>) -> tensor<ui64>
        %388 = "mhlo.multiply"(%387, %cst_6) {name = "multiply.1035"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %389 = "mhlo.convert"(%388) {name = "convert.1036"} : (tensor<ui64>) -> tensor<ui32>
        %390 = "mhlo.xor"(%385, %389) {name = "xor.1062"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %391 = "mhlo.xor"(%390, %13) {name = "xor.1063"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %392 = "mhlo.convert"(%391) {name = "convert.1070"} : (tensor<ui32>) -> tensor<ui64>
        %393 = "mhlo.multiply"(%392, %cst_7) {name = "multiply.1072"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %394 = "mhlo.shift_right_logical"(%393, %cst_8) {name = "shift-right-logical.1075"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %395 = "mhlo.convert"(%394) {name = "convert.1076"} : (tensor<ui64>) -> tensor<ui32>
        %396 = "mhlo.shift_right_logical"(%388, %cst_8) {name = "shift-right-logical.1038"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %397 = "mhlo.convert"(%396) {name = "convert.1039"} : (tensor<ui64>) -> tensor<ui32>
        %398 = "mhlo.shift_right_logical"(%369, %cst_8) {name = "shift-right-logical.1018"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %399 = "mhlo.convert"(%398) {name = "convert.1019"} : (tensor<ui64>) -> tensor<ui32>
        %400 = "mhlo.xor"(%397, %399) {name = "xor.1040"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %401 = "mhlo.xor"(%400, %cst_4) {name = "xor.1041"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %402 = "mhlo.convert"(%401) {name = "convert.1048"} : (tensor<ui32>) -> tensor<ui64>
        %403 = "mhlo.multiply"(%402, %cst_7) {name = "multiply.1050"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %404 = "mhlo.convert"(%403) {name = "convert.1051"} : (tensor<ui64>) -> tensor<ui32>
        %405 = "mhlo.xor"(%395, %404) {name = "xor.1086"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %406 = "mhlo.xor"(%405, %21) {name = "xor.1087"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %407 = "mhlo.convert"(%406) {name = "convert.1099"} : (tensor<ui32>) -> tensor<ui64>
        %408 = "mhlo.multiply"(%407, %cst_6) {name = "multiply.1101"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %409 = "mhlo.shift_right_logical"(%408, %cst_8) {name = "shift-right-logical.1104"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %410 = "mhlo.convert"(%409) {name = "convert.1105"} : (tensor<ui64>) -> tensor<ui32>
        %411 = "mhlo.shift_right_logical"(%403, %cst_8) {name = "shift-right-logical.1053"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %412 = "mhlo.convert"(%411) {name = "convert.1054"} : (tensor<ui64>) -> tensor<ui32>
        %413 = "mhlo.convert"(%372) {name = "convert.1029"} : (tensor<ui64>) -> tensor<ui32>
        %414 = "mhlo.xor"(%412, %413) {name = "xor.1064"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %415 = "mhlo.xor"(%414, %22) {name = "xor.1065"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %416 = "mhlo.convert"(%415) {name = "convert.1077"} : (tensor<ui32>) -> tensor<ui64>
        %417 = "mhlo.multiply"(%416, %cst_6) {name = "multiply.1079"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %418 = "mhlo.convert"(%417) {name = "convert.1080"} : (tensor<ui64>) -> tensor<ui32>
        %419 = "mhlo.xor"(%410, %418) {name = "xor.1106"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %420 = "mhlo.xor"(%419, %11) {name = "xor.1107"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %421 = "mhlo.convert"(%420) {name = "convert.1114"} : (tensor<ui32>) -> tensor<ui64>
        %422 = "mhlo.multiply"(%421, %cst_7) {name = "multiply.1116"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %423 = "mhlo.shift_right_logical"(%422, %cst_8) {name = "shift-right-logical.1119"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %424 = "mhlo.convert"(%423) {name = "convert.1120"} : (tensor<ui64>) -> tensor<ui32>
        %425 = "mhlo.shift_right_logical"(%417, %cst_8) {name = "shift-right-logical.1082"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %426 = "mhlo.convert"(%425) {name = "convert.1083"} : (tensor<ui64>) -> tensor<ui32>
        %427 = "mhlo.convert"(%383) {name = "convert.1058"} : (tensor<ui64>) -> tensor<ui32>
        %428 = "mhlo.xor"(%426, %427) {name = "xor.1084"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %429 = "mhlo.xor"(%428, %12) {name = "xor.1085"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %430 = "mhlo.convert"(%429) {name = "convert.1092"} : (tensor<ui32>) -> tensor<ui64>
        %431 = "mhlo.multiply"(%430, %cst_7) {name = "multiply.1094"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %432 = "mhlo.convert"(%431) {name = "convert.1095"} : (tensor<ui64>) -> tensor<ui32>
        %433 = "mhlo.xor"(%424, %432) {name = "xor.1130"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %434 = "mhlo.xor"(%433, %19) {name = "xor.1131"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %435 = "mhlo.convert"(%434) {name = "convert.1143"} : (tensor<ui32>) -> tensor<ui64>
        %436 = "mhlo.multiply"(%435, %cst_6) {name = "multiply.1145"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %437 = "mhlo.shift_right_logical"(%436, %cst_8) {name = "shift-right-logical.1148"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %438 = "mhlo.convert"(%437) {name = "convert.1149"} : (tensor<ui64>) -> tensor<ui32>
        %439 = "mhlo.shift_right_logical"(%431, %cst_8) {name = "shift-right-logical.1097"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %440 = "mhlo.convert"(%439) {name = "convert.1098"} : (tensor<ui64>) -> tensor<ui32>
        %441 = "mhlo.convert"(%393) {name = "convert.1073"} : (tensor<ui64>) -> tensor<ui32>
        %442 = "mhlo.xor"(%440, %441) {name = "xor.1108"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %443 = "mhlo.xor"(%442, %20) {name = "xor.1109"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %444 = "mhlo.convert"(%443) {name = "convert.1121"} : (tensor<ui32>) -> tensor<ui64>
        %445 = "mhlo.multiply"(%444, %cst_6) {name = "multiply.1123"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %446 = "mhlo.convert"(%445) {name = "convert.1124"} : (tensor<ui64>) -> tensor<ui32>
        %447 = "mhlo.xor"(%438, %446) {name = "xor.1150"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %448 = "mhlo.xor"(%447, %9) {name = "xor.1151"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %449 = "mhlo.convert"(%448) {name = "convert.1158"} : (tensor<ui32>) -> tensor<ui64>
        %450 = "mhlo.multiply"(%449, %cst_7) {name = "multiply.1160"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %451 = "mhlo.shift_right_logical"(%450, %cst_8) {name = "shift-right-logical.1163"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %452 = "mhlo.convert"(%451) {name = "convert.1164"} : (tensor<ui64>) -> tensor<ui32>
        %453 = "mhlo.shift_right_logical"(%445, %cst_8) {name = "shift-right-logical.1126"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %454 = "mhlo.convert"(%453) {name = "convert.1127"} : (tensor<ui64>) -> tensor<ui32>
        %455 = "mhlo.convert"(%408) {name = "convert.1102"} : (tensor<ui64>) -> tensor<ui32>
        %456 = "mhlo.xor"(%454, %455) {name = "xor.1128"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %457 = "mhlo.xor"(%456, %10) {name = "xor.1129"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %458 = "mhlo.convert"(%457) {name = "convert.1136"} : (tensor<ui32>) -> tensor<ui64>
        %459 = "mhlo.multiply"(%458, %cst_7) {name = "multiply.1138"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %460 = "mhlo.convert"(%459) {name = "convert.1139"} : (tensor<ui64>) -> tensor<ui32>
        %461 = "mhlo.xor"(%452, %460) {name = "xor.1174"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %462 = "mhlo.xor"(%461, %17) {name = "xor.1175"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %463 = "mhlo.convert"(%462) {name = "convert.1187"} : (tensor<ui32>) -> tensor<ui64>
        %464 = "mhlo.multiply"(%463, %cst_6) {name = "multiply.1189"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %465 = "mhlo.shift_right_logical"(%464, %cst_8) {name = "shift-right-logical.1192"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %466 = "mhlo.convert"(%465) {name = "convert.1193"} : (tensor<ui64>) -> tensor<ui32>
        %467 = "mhlo.shift_right_logical"(%459, %cst_8) {name = "shift-right-logical.1141"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %468 = "mhlo.convert"(%467) {name = "convert.1142"} : (tensor<ui64>) -> tensor<ui32>
        %469 = "mhlo.convert"(%422) {name = "convert.1117"} : (tensor<ui64>) -> tensor<ui32>
        %470 = "mhlo.xor"(%468, %469) {name = "xor.1152"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %471 = "mhlo.xor"(%470, %18) {name = "xor.1153"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %472 = "mhlo.convert"(%471) {name = "convert.1165"} : (tensor<ui32>) -> tensor<ui64>
        %473 = "mhlo.multiply"(%472, %cst_6) {name = "multiply.1167"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %474 = "mhlo.convert"(%473) {name = "convert.1168"} : (tensor<ui64>) -> tensor<ui32>
        %475 = "mhlo.xor"(%466, %474) {name = "xor.1194"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %476 = "mhlo.xor"(%475, %7) {name = "xor.1195"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %477 = "mhlo.convert"(%476) {name = "convert.1202"} : (tensor<ui32>) -> tensor<ui64>
        %478 = "mhlo.multiply"(%477, %cst_7) {name = "multiply.1204"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %479 = "mhlo.shift_right_logical"(%478, %cst_8) {name = "shift-right-logical.1207"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %480 = "mhlo.convert"(%479) {name = "convert.1208"} : (tensor<ui64>) -> tensor<ui32>
        %481 = "mhlo.shift_right_logical"(%473, %cst_8) {name = "shift-right-logical.1170"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %482 = "mhlo.convert"(%481) {name = "convert.1171"} : (tensor<ui64>) -> tensor<ui32>
        %483 = "mhlo.convert"(%436) {name = "convert.1146"} : (tensor<ui64>) -> tensor<ui32>
        %484 = "mhlo.xor"(%482, %483) {name = "xor.1172"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %485 = "mhlo.xor"(%484, %8) {name = "xor.1173"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %486 = "mhlo.convert"(%485) {name = "convert.1180"} : (tensor<ui32>) -> tensor<ui64>
        %487 = "mhlo.multiply"(%486, %cst_7) {name = "multiply.1182"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %488 = "mhlo.convert"(%487) {name = "convert.1183"} : (tensor<ui64>) -> tensor<ui32>
        %489 = "mhlo.xor"(%480, %488) {name = "xor.1218"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %490 = "mhlo.xor"(%489, %15) {name = "xor.1219"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %491 = "mhlo.convert"(%490) {name = "convert.1231"} : (tensor<ui32>) -> tensor<ui64>
        %492 = "mhlo.multiply"(%491, %cst_6) {name = "multiply.1233"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %493 = "mhlo.shift_right_logical"(%492, %cst_8) {name = "shift-right-logical.1236"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %494 = "mhlo.convert"(%493) {name = "convert.1237"} : (tensor<ui64>) -> tensor<ui32>
        %495 = "mhlo.shift_right_logical"(%487, %cst_8) {name = "shift-right-logical.1185"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %496 = "mhlo.convert"(%495) {name = "convert.1186"} : (tensor<ui64>) -> tensor<ui32>
        %497 = "mhlo.convert"(%450) {name = "convert.1161"} : (tensor<ui64>) -> tensor<ui32>
        %498 = "mhlo.xor"(%496, %497) {name = "xor.1196"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %499 = "mhlo.xor"(%498, %16) {name = "xor.1197"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %500 = "mhlo.convert"(%499) {name = "convert.1209"} : (tensor<ui32>) -> tensor<ui64>
        %501 = "mhlo.multiply"(%500, %cst_6) {name = "multiply.1211"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %502 = "mhlo.convert"(%501) {name = "convert.1212"} : (tensor<ui64>) -> tensor<ui32>
        %503 = "mhlo.xor"(%494, %502) {name = "xor.1238"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %504 = "mhlo.xor"(%503, %5) {name = "xor.1239"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %505 = "mhlo.convert"(%504) {name = "convert.1260"} : (tensor<ui32>) -> tensor<ui64>
        %506 = "mhlo.convert"(%492) {name = "convert.1234"} : (tensor<ui64>) -> tensor<ui32>
        %507 = "mhlo.convert"(%506) {name = "convert.1261"} : (tensor<ui32>) -> tensor<ui64>
        %508 = "mhlo.shift_left"(%507, %cst_8) {name = "shift-left.1263"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %509 = "mhlo.or"(%505, %508) {name = "or.1264"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %510 = "mhlo.reshape"(%509) {name = "reshape.1265"} : (tensor<ui64>) -> tensor<1xui64>
        %511 = "mhlo.shift_left"(%23, %cst_8) {name = "shift-left.1250"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %512 = "mhlo.or"(%511, %23) {name = "or.1251"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %513 = "mhlo.reshape"(%512) {name = "reshape.1257"} : (tensor<ui64>) -> tensor<1xui64>
        %514 = "mhlo.shift_right_logical"(%501, %cst_8) {name = "shift-right-logical.1214"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %515 = "mhlo.convert"(%514) {name = "convert.1215"} : (tensor<ui64>) -> tensor<ui32>
        %516 = "mhlo.convert"(%464) {name = "convert.1190"} : (tensor<ui64>) -> tensor<ui32>
        %517 = "mhlo.xor"(%515, %516) {name = "xor.1216"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %518 = "mhlo.xor"(%517, %6) {name = "xor.1217"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %519 = "mhlo.convert"(%518) {name = "convert.1224"} : (tensor<ui32>) -> tensor<ui64>
        %520 = "mhlo.multiply"(%519, %cst_7) {name = "multiply.1226"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %521 = "mhlo.shift_right_logical"(%520, %cst_8) {name = "shift-right-logical.1229"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %522 = "mhlo.convert"(%521) {name = "convert.1230"} : (tensor<ui64>) -> tensor<ui32>
        %523 = "mhlo.convert"(%478) {name = "convert.1205"} : (tensor<ui64>) -> tensor<ui32>
        %524 = "mhlo.xor"(%522, %523) {name = "xor.1240"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %525 = "mhlo.xor"(%524, %14) {name = "xor.1241"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %526 = "mhlo.convert"(%525) {name = "convert.1252"} : (tensor<ui32>) -> tensor<ui64>
        %527 = "mhlo.convert"(%520) {name = "convert.1227"} : (tensor<ui64>) -> tensor<ui32>
        %528 = "mhlo.convert"(%527) {name = "convert.1253"} : (tensor<ui32>) -> tensor<ui64>
        %529 = "mhlo.shift_left"(%528, %cst_8) {name = "shift-left.1255"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %530 = "mhlo.or"(%526, %529) {name = "or.1256"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %531 = "mhlo.reshape"(%530) {name = "reshape.1258"} : (tensor<ui64>) -> tensor<1xui64>
        %532 = "mhlo.concatenate"(%513, %531) {dimension = 0 : i64} : (tensor<1xui64>, tensor<1xui64>) -> tensor<2xui64>
        %533 = "mhlo.concatenate"(%510, %532) {dimension = 0 : i64} : (tensor<1xui64>, tensor<2xui64>) -> tensor<3xui64>
        %534 = "mhlo.rng_bit_generator"(%533) {rng_algorithm = 2 : i32} : (tensor<3xui64>) -> tuple<tensor<3xui64>, tensor<2x2xui32>>
        %535 = "mhlo.get_tuple_element"(%534) {index = 1 : i32, name = "get-tuple-element.1268"} : (tuple<tensor<3xui64>, tensor<2x2xui32>>) -> tensor<2x2xui32>
        %536 = "mhlo.bitcast_convert"(%535) {name = "bitcast-convert.1270"} : (tensor<2x2xui32>) -> tensor<2x2xi32>
        %537 = "mhlo.slice"(%536) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xi32>) -> tensor<1x2xi32>
        %538 = "mhlo.reshape"(%537) {name = "reshape.1272"} : (tensor<1x2xi32>) -> tensor<2xi32>
        %539 = "mhlo.slice"(%538) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %540 = "mhlo.reshape"(%539) {name = "reshape.1276"} : (tensor<1xi32>) -> tensor<i32>
        %541 = "mhlo.convert"(%540) {name = "convert.1279"} : (tensor<i32>) -> tensor<ui64>
        %542 = "mhlo.slice"(%538) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %543 = "mhlo.reshape"(%542) {name = "reshape.1278"} : (tensor<1xi32>) -> tensor<i32>
        %544 = "mhlo.convert"(%543) {name = "convert.1280"} : (tensor<i32>) -> tensor<ui64>
        %545 = "mhlo.shift_left"(%544, %cst_8) {name = "shift-left.1282"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %546 = "mhlo.or"(%541, %545) {name = "or.1283"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %547 = "mhlo.convert"(%546) {name = "convert.1286"} : (tensor<ui64>) -> tensor<ui32>
        %548 = "mhlo.convert"(%547) {name = "convert.1289"} : (tensor<ui32>) -> tensor<ui64>
        %549 = "mhlo.convert"(%548) {name = "convert.1291"} : (tensor<ui64>) -> tensor<ui32>
        %550 = "mhlo.convert"(%549) {name = "convert.1301"} : (tensor<ui32>) -> tensor<ui64>
        %551 = "mhlo.multiply"(%550, %cst_7) {name = "multiply.1303"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %552 = "mhlo.shift_right_logical"(%551, %cst_8) {name = "shift-right-logical.1306"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %553 = "mhlo.convert"(%552) {name = "convert.1307"} : (tensor<ui64>) -> tensor<ui32>
        %554 = "mhlo.shift_right_logical"(%546, %cst_8) {name = "shift-right-logical.1287"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %555 = "mhlo.convert"(%554) {name = "convert.1288"} : (tensor<ui64>) -> tensor<ui32>
        %556 = "mhlo.convert"(%555) {name = "convert.1290"} : (tensor<ui32>) -> tensor<ui64>
        %557 = "mhlo.shift_right_logical"(%556, %cst_8) {name = "shift-right-logical.1297"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %558 = "mhlo.convert"(%557) {name = "convert.1298"} : (tensor<ui64>) -> tensor<ui32>
        %559 = "mhlo.xor"(%553, %558) {name = "xor.1317"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %560 = "mhlo.xor"(%559, %cst_5) {name = "xor.1318"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %561 = "mhlo.convert"(%560) {name = "convert.1330"} : (tensor<ui32>) -> tensor<ui64>
        %562 = "mhlo.multiply"(%561, %cst_6) {name = "multiply.1332"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %563 = "mhlo.shift_right_logical"(%562, %cst_8) {name = "shift-right-logical.1335"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %564 = "mhlo.convert"(%563) {name = "convert.1336"} : (tensor<ui64>) -> tensor<ui32>
        %565 = "mhlo.convert"(%556) {name = "convert.1295"} : (tensor<ui64>) -> tensor<ui32>
        %566 = "mhlo.convert"(%565) {name = "convert.1308"} : (tensor<ui32>) -> tensor<ui64>
        %567 = "mhlo.multiply"(%566, %cst_6) {name = "multiply.1310"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %568 = "mhlo.convert"(%567) {name = "convert.1311"} : (tensor<ui64>) -> tensor<ui32>
        %569 = "mhlo.xor"(%564, %568) {name = "xor.1337"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %570 = "mhlo.xor"(%569, %13) {name = "xor.1338"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %571 = "mhlo.convert"(%570) {name = "convert.1345"} : (tensor<ui32>) -> tensor<ui64>
        %572 = "mhlo.multiply"(%571, %cst_7) {name = "multiply.1347"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %573 = "mhlo.shift_right_logical"(%572, %cst_8) {name = "shift-right-logical.1350"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %574 = "mhlo.convert"(%573) {name = "convert.1351"} : (tensor<ui64>) -> tensor<ui32>
        %575 = "mhlo.shift_right_logical"(%567, %cst_8) {name = "shift-right-logical.1313"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %576 = "mhlo.convert"(%575) {name = "convert.1314"} : (tensor<ui64>) -> tensor<ui32>
        %577 = "mhlo.shift_right_logical"(%548, %cst_8) {name = "shift-right-logical.1293"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %578 = "mhlo.convert"(%577) {name = "convert.1294"} : (tensor<ui64>) -> tensor<ui32>
        %579 = "mhlo.xor"(%576, %578) {name = "xor.1315"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %580 = "mhlo.xor"(%579, %cst_4) {name = "xor.1316"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %581 = "mhlo.convert"(%580) {name = "convert.1323"} : (tensor<ui32>) -> tensor<ui64>
        %582 = "mhlo.multiply"(%581, %cst_7) {name = "multiply.1325"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %583 = "mhlo.convert"(%582) {name = "convert.1326"} : (tensor<ui64>) -> tensor<ui32>
        %584 = "mhlo.xor"(%574, %583) {name = "xor.1361"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %585 = "mhlo.xor"(%584, %21) {name = "xor.1362"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %586 = "mhlo.convert"(%585) {name = "convert.1374"} : (tensor<ui32>) -> tensor<ui64>
        %587 = "mhlo.multiply"(%586, %cst_6) {name = "multiply.1376"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %588 = "mhlo.shift_right_logical"(%587, %cst_8) {name = "shift-right-logical.1379"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %589 = "mhlo.convert"(%588) {name = "convert.1380"} : (tensor<ui64>) -> tensor<ui32>
        %590 = "mhlo.shift_right_logical"(%582, %cst_8) {name = "shift-right-logical.1328"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %591 = "mhlo.convert"(%590) {name = "convert.1329"} : (tensor<ui64>) -> tensor<ui32>
        %592 = "mhlo.convert"(%551) {name = "convert.1304"} : (tensor<ui64>) -> tensor<ui32>
        %593 = "mhlo.xor"(%591, %592) {name = "xor.1339"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %594 = "mhlo.xor"(%593, %22) {name = "xor.1340"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %595 = "mhlo.convert"(%594) {name = "convert.1352"} : (tensor<ui32>) -> tensor<ui64>
        %596 = "mhlo.multiply"(%595, %cst_6) {name = "multiply.1354"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %597 = "mhlo.convert"(%596) {name = "convert.1355"} : (tensor<ui64>) -> tensor<ui32>
        %598 = "mhlo.xor"(%589, %597) {name = "xor.1381"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %599 = "mhlo.xor"(%598, %11) {name = "xor.1382"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %600 = "mhlo.convert"(%599) {name = "convert.1389"} : (tensor<ui32>) -> tensor<ui64>
        %601 = "mhlo.multiply"(%600, %cst_7) {name = "multiply.1391"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %602 = "mhlo.shift_right_logical"(%601, %cst_8) {name = "shift-right-logical.1394"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %603 = "mhlo.convert"(%602) {name = "convert.1395"} : (tensor<ui64>) -> tensor<ui32>
        %604 = "mhlo.shift_right_logical"(%596, %cst_8) {name = "shift-right-logical.1357"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %605 = "mhlo.convert"(%604) {name = "convert.1358"} : (tensor<ui64>) -> tensor<ui32>
        %606 = "mhlo.convert"(%562) {name = "convert.1333"} : (tensor<ui64>) -> tensor<ui32>
        %607 = "mhlo.xor"(%605, %606) {name = "xor.1359"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %608 = "mhlo.xor"(%607, %12) {name = "xor.1360"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %609 = "mhlo.convert"(%608) {name = "convert.1367"} : (tensor<ui32>) -> tensor<ui64>
        %610 = "mhlo.multiply"(%609, %cst_7) {name = "multiply.1369"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %611 = "mhlo.convert"(%610) {name = "convert.1370"} : (tensor<ui64>) -> tensor<ui32>
        %612 = "mhlo.xor"(%603, %611) {name = "xor.1405"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %613 = "mhlo.xor"(%612, %19) {name = "xor.1406"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %614 = "mhlo.convert"(%613) {name = "convert.1418"} : (tensor<ui32>) -> tensor<ui64>
        %615 = "mhlo.multiply"(%614, %cst_6) {name = "multiply.1420"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %616 = "mhlo.shift_right_logical"(%615, %cst_8) {name = "shift-right-logical.1423"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %617 = "mhlo.convert"(%616) {name = "convert.1424"} : (tensor<ui64>) -> tensor<ui32>
        %618 = "mhlo.shift_right_logical"(%610, %cst_8) {name = "shift-right-logical.1372"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %619 = "mhlo.convert"(%618) {name = "convert.1373"} : (tensor<ui64>) -> tensor<ui32>
        %620 = "mhlo.convert"(%572) {name = "convert.1348"} : (tensor<ui64>) -> tensor<ui32>
        %621 = "mhlo.xor"(%619, %620) {name = "xor.1383"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %622 = "mhlo.xor"(%621, %20) {name = "xor.1384"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %623 = "mhlo.convert"(%622) {name = "convert.1396"} : (tensor<ui32>) -> tensor<ui64>
        %624 = "mhlo.multiply"(%623, %cst_6) {name = "multiply.1398"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %625 = "mhlo.convert"(%624) {name = "convert.1399"} : (tensor<ui64>) -> tensor<ui32>
        %626 = "mhlo.xor"(%617, %625) {name = "xor.1425"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %627 = "mhlo.xor"(%626, %9) {name = "xor.1426"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %628 = "mhlo.convert"(%627) {name = "convert.1433"} : (tensor<ui32>) -> tensor<ui64>
        %629 = "mhlo.multiply"(%628, %cst_7) {name = "multiply.1435"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %630 = "mhlo.shift_right_logical"(%629, %cst_8) {name = "shift-right-logical.1438"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %631 = "mhlo.convert"(%630) {name = "convert.1439"} : (tensor<ui64>) -> tensor<ui32>
        %632 = "mhlo.shift_right_logical"(%624, %cst_8) {name = "shift-right-logical.1401"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %633 = "mhlo.convert"(%632) {name = "convert.1402"} : (tensor<ui64>) -> tensor<ui32>
        %634 = "mhlo.convert"(%587) {name = "convert.1377"} : (tensor<ui64>) -> tensor<ui32>
        %635 = "mhlo.xor"(%633, %634) {name = "xor.1403"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %636 = "mhlo.xor"(%635, %10) {name = "xor.1404"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %637 = "mhlo.convert"(%636) {name = "convert.1411"} : (tensor<ui32>) -> tensor<ui64>
        %638 = "mhlo.multiply"(%637, %cst_7) {name = "multiply.1413"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %639 = "mhlo.convert"(%638) {name = "convert.1414"} : (tensor<ui64>) -> tensor<ui32>
        %640 = "mhlo.xor"(%631, %639) {name = "xor.1449"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %641 = "mhlo.xor"(%640, %17) {name = "xor.1450"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %642 = "mhlo.convert"(%641) {name = "convert.1462"} : (tensor<ui32>) -> tensor<ui64>
        %643 = "mhlo.multiply"(%642, %cst_6) {name = "multiply.1464"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %644 = "mhlo.shift_right_logical"(%643, %cst_8) {name = "shift-right-logical.1467"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %645 = "mhlo.convert"(%644) {name = "convert.1468"} : (tensor<ui64>) -> tensor<ui32>
        %646 = "mhlo.shift_right_logical"(%638, %cst_8) {name = "shift-right-logical.1416"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %647 = "mhlo.convert"(%646) {name = "convert.1417"} : (tensor<ui64>) -> tensor<ui32>
        %648 = "mhlo.convert"(%601) {name = "convert.1392"} : (tensor<ui64>) -> tensor<ui32>
        %649 = "mhlo.xor"(%647, %648) {name = "xor.1427"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %650 = "mhlo.xor"(%649, %18) {name = "xor.1428"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %651 = "mhlo.convert"(%650) {name = "convert.1440"} : (tensor<ui32>) -> tensor<ui64>
        %652 = "mhlo.multiply"(%651, %cst_6) {name = "multiply.1442"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %653 = "mhlo.convert"(%652) {name = "convert.1443"} : (tensor<ui64>) -> tensor<ui32>
        %654 = "mhlo.xor"(%645, %653) {name = "xor.1469"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %655 = "mhlo.xor"(%654, %7) {name = "xor.1470"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %656 = "mhlo.convert"(%655) {name = "convert.1477"} : (tensor<ui32>) -> tensor<ui64>
        %657 = "mhlo.multiply"(%656, %cst_7) {name = "multiply.1479"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %658 = "mhlo.shift_right_logical"(%657, %cst_8) {name = "shift-right-logical.1482"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %659 = "mhlo.convert"(%658) {name = "convert.1483"} : (tensor<ui64>) -> tensor<ui32>
        %660 = "mhlo.shift_right_logical"(%652, %cst_8) {name = "shift-right-logical.1445"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %661 = "mhlo.convert"(%660) {name = "convert.1446"} : (tensor<ui64>) -> tensor<ui32>
        %662 = "mhlo.convert"(%615) {name = "convert.1421"} : (tensor<ui64>) -> tensor<ui32>
        %663 = "mhlo.xor"(%661, %662) {name = "xor.1447"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %664 = "mhlo.xor"(%663, %8) {name = "xor.1448"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %665 = "mhlo.convert"(%664) {name = "convert.1455"} : (tensor<ui32>) -> tensor<ui64>
        %666 = "mhlo.multiply"(%665, %cst_7) {name = "multiply.1457"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %667 = "mhlo.convert"(%666) {name = "convert.1458"} : (tensor<ui64>) -> tensor<ui32>
        %668 = "mhlo.xor"(%659, %667) {name = "xor.1493"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %669 = "mhlo.xor"(%668, %15) {name = "xor.1494"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %670 = "mhlo.convert"(%669) {name = "convert.1506"} : (tensor<ui32>) -> tensor<ui64>
        %671 = "mhlo.multiply"(%670, %cst_6) {name = "multiply.1508"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %672 = "mhlo.shift_right_logical"(%671, %cst_8) {name = "shift-right-logical.1511"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %673 = "mhlo.convert"(%672) {name = "convert.1512"} : (tensor<ui64>) -> tensor<ui32>
        %674 = "mhlo.shift_right_logical"(%666, %cst_8) {name = "shift-right-logical.1460"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %675 = "mhlo.convert"(%674) {name = "convert.1461"} : (tensor<ui64>) -> tensor<ui32>
        %676 = "mhlo.convert"(%629) {name = "convert.1436"} : (tensor<ui64>) -> tensor<ui32>
        %677 = "mhlo.xor"(%675, %676) {name = "xor.1471"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %678 = "mhlo.xor"(%677, %16) {name = "xor.1472"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %679 = "mhlo.convert"(%678) {name = "convert.1484"} : (tensor<ui32>) -> tensor<ui64>
        %680 = "mhlo.multiply"(%679, %cst_6) {name = "multiply.1486"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %681 = "mhlo.convert"(%680) {name = "convert.1487"} : (tensor<ui64>) -> tensor<ui32>
        %682 = "mhlo.xor"(%673, %681) {name = "xor.1513"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %683 = "mhlo.xor"(%682, %5) {name = "xor.1514"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %684 = "mhlo.convert"(%683) {name = "convert.1535"} : (tensor<ui32>) -> tensor<ui64>
        %685 = "mhlo.convert"(%671) {name = "convert.1509"} : (tensor<ui64>) -> tensor<ui32>
        %686 = "mhlo.convert"(%685) {name = "convert.1536"} : (tensor<ui32>) -> tensor<ui64>
        %687 = "mhlo.shift_left"(%686, %cst_8) {name = "shift-left.1538"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %688 = "mhlo.or"(%684, %687) {name = "or.1539"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %689 = "mhlo.reshape"(%688) {name = "reshape.1540"} : (tensor<ui64>) -> tensor<1xui64>
        %690 = "mhlo.shift_left"(%23, %cst_8) {name = "shift-left.1525"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %691 = "mhlo.or"(%690, %23) {name = "or.1526"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %692 = "mhlo.reshape"(%691) {name = "reshape.1532"} : (tensor<ui64>) -> tensor<1xui64>
        %693 = "mhlo.shift_right_logical"(%680, %cst_8) {name = "shift-right-logical.1489"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %694 = "mhlo.convert"(%693) {name = "convert.1490"} : (tensor<ui64>) -> tensor<ui32>
        %695 = "mhlo.convert"(%643) {name = "convert.1465"} : (tensor<ui64>) -> tensor<ui32>
        %696 = "mhlo.xor"(%694, %695) {name = "xor.1491"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %697 = "mhlo.xor"(%696, %6) {name = "xor.1492"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %698 = "mhlo.convert"(%697) {name = "convert.1499"} : (tensor<ui32>) -> tensor<ui64>
        %699 = "mhlo.multiply"(%698, %cst_7) {name = "multiply.1501"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %700 = "mhlo.shift_right_logical"(%699, %cst_8) {name = "shift-right-logical.1504"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %701 = "mhlo.convert"(%700) {name = "convert.1505"} : (tensor<ui64>) -> tensor<ui32>
        %702 = "mhlo.convert"(%657) {name = "convert.1480"} : (tensor<ui64>) -> tensor<ui32>
        %703 = "mhlo.xor"(%701, %702) {name = "xor.1515"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %704 = "mhlo.xor"(%703, %14) {name = "xor.1516"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %705 = "mhlo.convert"(%704) {name = "convert.1527"} : (tensor<ui32>) -> tensor<ui64>
        %706 = "mhlo.convert"(%699) {name = "convert.1502"} : (tensor<ui64>) -> tensor<ui32>
        %707 = "mhlo.convert"(%706) {name = "convert.1528"} : (tensor<ui32>) -> tensor<ui64>
        %708 = "mhlo.shift_left"(%707, %cst_8) {name = "shift-left.1530"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %709 = "mhlo.or"(%705, %708) {name = "or.1531"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %710 = "mhlo.reshape"(%709) {name = "reshape.1533"} : (tensor<ui64>) -> tensor<1xui64>
        %711 = "mhlo.concatenate"(%692, %710) {dimension = 0 : i64} : (tensor<1xui64>, tensor<1xui64>) -> tensor<2xui64>
        %712 = "mhlo.concatenate"(%689, %711) {dimension = 0 : i64} : (tensor<1xui64>, tensor<2xui64>) -> tensor<3xui64>
        %713 = "mhlo.rng_bit_generator"(%712) {rng_algorithm = 2 : i32} : (tensor<3xui64>) -> tuple<tensor<3xui64>, tensor<1x2xui32>>
        %714 = "mhlo.get_tuple_element"(%713) {index = 1 : i32, name = "get-tuple-element.1543"} : (tuple<tensor<3xui64>, tensor<1x2xui32>>) -> tensor<1x2xui32>
        %715 = "mhlo.bitcast_convert"(%714) {name = "bitcast-convert.1545"} : (tensor<1x2xui32>) -> tensor<1x2xi32>
        %716 = "mhlo.reshape"(%715) {name = "reshape.1547"} : (tensor<1x2xi32>) -> tensor<2xi32>
        %717 = "mhlo.slice"(%716) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %718 = "mhlo.reshape"(%717) {name = "reshape.1549"} : (tensor<1xi32>) -> tensor<i32>
        %719 = "mhlo.convert"(%718) {name = "convert.1553"} : (tensor<i32>) -> tensor<ui64>
        %720 = "mhlo.slice"(%716) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %721 = "mhlo.reshape"(%720) {name = "reshape.1551"} : (tensor<1xi32>) -> tensor<i32>
        %722 = "mhlo.convert"(%721) {name = "convert.1554"} : (tensor<i32>) -> tensor<ui64>
        %723 = "mhlo.shift_left"(%722, %cst_8) {name = "shift-left.1556"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %724 = "mhlo.or"(%719, %723) {name = "or.1557"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %725 = "mhlo.convert"(%724) {name = "convert.1561"} : (tensor<ui64>) -> tensor<ui32>
        %726 = "mhlo.convert"(%725) {name = "convert.1564"} : (tensor<ui32>) -> tensor<ui64>
        %727 = "mhlo.convert"(%726) {name = "convert.1566"} : (tensor<ui64>) -> tensor<ui32>
        %728 = "mhlo.convert"(%727) {name = "convert.1576"} : (tensor<ui32>) -> tensor<ui64>
        %729 = "mhlo.multiply"(%728, %cst_7) {name = "multiply.1578"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %730 = "mhlo.shift_right_logical"(%729, %cst_8) {name = "shift-right-logical.1581"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %731 = "mhlo.convert"(%730) {name = "convert.1582"} : (tensor<ui64>) -> tensor<ui32>
        %732 = "mhlo.shift_right_logical"(%724, %cst_8) {name = "shift-right-logical.1562"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %733 = "mhlo.convert"(%732) {name = "convert.1563"} : (tensor<ui64>) -> tensor<ui32>
        %734 = "mhlo.convert"(%733) {name = "convert.1565"} : (tensor<ui32>) -> tensor<ui64>
        %735 = "mhlo.shift_right_logical"(%734, %cst_8) {name = "shift-right-logical.1572"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %736 = "mhlo.convert"(%735) {name = "convert.1573"} : (tensor<ui64>) -> tensor<ui32>
        %737 = "mhlo.xor"(%731, %736) {name = "xor.1592"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %738 = "mhlo.xor"(%737, %cst_5) {name = "xor.1593"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %739 = "mhlo.convert"(%738) {name = "convert.1605"} : (tensor<ui32>) -> tensor<ui64>
        %740 = "mhlo.multiply"(%739, %cst_6) {name = "multiply.1607"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %741 = "mhlo.shift_right_logical"(%740, %cst_8) {name = "shift-right-logical.1610"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %742 = "mhlo.convert"(%741) {name = "convert.1611"} : (tensor<ui64>) -> tensor<ui32>
        %743 = "mhlo.convert"(%734) {name = "convert.1570"} : (tensor<ui64>) -> tensor<ui32>
        %744 = "mhlo.convert"(%743) {name = "convert.1583"} : (tensor<ui32>) -> tensor<ui64>
        %745 = "mhlo.multiply"(%744, %cst_6) {name = "multiply.1585"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %746 = "mhlo.convert"(%745) {name = "convert.1586"} : (tensor<ui64>) -> tensor<ui32>
        %747 = "mhlo.xor"(%742, %746) {name = "xor.1612"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %748 = "mhlo.xor"(%747, %13) {name = "xor.1613"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %749 = "mhlo.convert"(%748) {name = "convert.1620"} : (tensor<ui32>) -> tensor<ui64>
        %750 = "mhlo.multiply"(%749, %cst_7) {name = "multiply.1622"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %751 = "mhlo.shift_right_logical"(%750, %cst_8) {name = "shift-right-logical.1625"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %752 = "mhlo.convert"(%751) {name = "convert.1626"} : (tensor<ui64>) -> tensor<ui32>
        %753 = "mhlo.shift_right_logical"(%745, %cst_8) {name = "shift-right-logical.1588"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %754 = "mhlo.convert"(%753) {name = "convert.1589"} : (tensor<ui64>) -> tensor<ui32>
        %755 = "mhlo.shift_right_logical"(%726, %cst_8) {name = "shift-right-logical.1568"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %756 = "mhlo.convert"(%755) {name = "convert.1569"} : (tensor<ui64>) -> tensor<ui32>
        %757 = "mhlo.xor"(%754, %756) {name = "xor.1590"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %758 = "mhlo.xor"(%757, %cst_4) {name = "xor.1591"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %759 = "mhlo.convert"(%758) {name = "convert.1598"} : (tensor<ui32>) -> tensor<ui64>
        %760 = "mhlo.multiply"(%759, %cst_7) {name = "multiply.1600"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %761 = "mhlo.convert"(%760) {name = "convert.1601"} : (tensor<ui64>) -> tensor<ui32>
        %762 = "mhlo.xor"(%752, %761) {name = "xor.1636"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %763 = "mhlo.xor"(%762, %21) {name = "xor.1637"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %764 = "mhlo.convert"(%763) {name = "convert.1649"} : (tensor<ui32>) -> tensor<ui64>
        %765 = "mhlo.multiply"(%764, %cst_6) {name = "multiply.1651"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %766 = "mhlo.shift_right_logical"(%765, %cst_8) {name = "shift-right-logical.1654"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %767 = "mhlo.convert"(%766) {name = "convert.1655"} : (tensor<ui64>) -> tensor<ui32>
        %768 = "mhlo.shift_right_logical"(%760, %cst_8) {name = "shift-right-logical.1603"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %769 = "mhlo.convert"(%768) {name = "convert.1604"} : (tensor<ui64>) -> tensor<ui32>
        %770 = "mhlo.convert"(%729) {name = "convert.1579"} : (tensor<ui64>) -> tensor<ui32>
        %771 = "mhlo.xor"(%769, %770) {name = "xor.1614"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %772 = "mhlo.xor"(%771, %22) {name = "xor.1615"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %773 = "mhlo.convert"(%772) {name = "convert.1627"} : (tensor<ui32>) -> tensor<ui64>
        %774 = "mhlo.multiply"(%773, %cst_6) {name = "multiply.1629"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %775 = "mhlo.convert"(%774) {name = "convert.1630"} : (tensor<ui64>) -> tensor<ui32>
        %776 = "mhlo.xor"(%767, %775) {name = "xor.1656"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %777 = "mhlo.xor"(%776, %11) {name = "xor.1657"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %778 = "mhlo.convert"(%777) {name = "convert.1664"} : (tensor<ui32>) -> tensor<ui64>
        %779 = "mhlo.multiply"(%778, %cst_7) {name = "multiply.1666"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %780 = "mhlo.shift_right_logical"(%779, %cst_8) {name = "shift-right-logical.1669"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %781 = "mhlo.convert"(%780) {name = "convert.1670"} : (tensor<ui64>) -> tensor<ui32>
        %782 = "mhlo.shift_right_logical"(%774, %cst_8) {name = "shift-right-logical.1632"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %783 = "mhlo.convert"(%782) {name = "convert.1633"} : (tensor<ui64>) -> tensor<ui32>
        %784 = "mhlo.convert"(%740) {name = "convert.1608"} : (tensor<ui64>) -> tensor<ui32>
        %785 = "mhlo.xor"(%783, %784) {name = "xor.1634"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %786 = "mhlo.xor"(%785, %12) {name = "xor.1635"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %787 = "mhlo.convert"(%786) {name = "convert.1642"} : (tensor<ui32>) -> tensor<ui64>
        %788 = "mhlo.multiply"(%787, %cst_7) {name = "multiply.1644"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %789 = "mhlo.convert"(%788) {name = "convert.1645"} : (tensor<ui64>) -> tensor<ui32>
        %790 = "mhlo.xor"(%781, %789) {name = "xor.1680"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %791 = "mhlo.xor"(%790, %19) {name = "xor.1681"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %792 = "mhlo.convert"(%791) {name = "convert.1693"} : (tensor<ui32>) -> tensor<ui64>
        %793 = "mhlo.multiply"(%792, %cst_6) {name = "multiply.1695"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %794 = "mhlo.shift_right_logical"(%793, %cst_8) {name = "shift-right-logical.1698"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %795 = "mhlo.convert"(%794) {name = "convert.1699"} : (tensor<ui64>) -> tensor<ui32>
        %796 = "mhlo.shift_right_logical"(%788, %cst_8) {name = "shift-right-logical.1647"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %797 = "mhlo.convert"(%796) {name = "convert.1648"} : (tensor<ui64>) -> tensor<ui32>
        %798 = "mhlo.convert"(%750) {name = "convert.1623"} : (tensor<ui64>) -> tensor<ui32>
        %799 = "mhlo.xor"(%797, %798) {name = "xor.1658"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %800 = "mhlo.xor"(%799, %20) {name = "xor.1659"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %801 = "mhlo.convert"(%800) {name = "convert.1671"} : (tensor<ui32>) -> tensor<ui64>
        %802 = "mhlo.multiply"(%801, %cst_6) {name = "multiply.1673"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %803 = "mhlo.convert"(%802) {name = "convert.1674"} : (tensor<ui64>) -> tensor<ui32>
        %804 = "mhlo.xor"(%795, %803) {name = "xor.1700"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %805 = "mhlo.xor"(%804, %9) {name = "xor.1701"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %806 = "mhlo.convert"(%805) {name = "convert.1708"} : (tensor<ui32>) -> tensor<ui64>
        %807 = "mhlo.multiply"(%806, %cst_7) {name = "multiply.1710"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %808 = "mhlo.shift_right_logical"(%807, %cst_8) {name = "shift-right-logical.1713"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %809 = "mhlo.convert"(%808) {name = "convert.1714"} : (tensor<ui64>) -> tensor<ui32>
        %810 = "mhlo.shift_right_logical"(%802, %cst_8) {name = "shift-right-logical.1676"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %811 = "mhlo.convert"(%810) {name = "convert.1677"} : (tensor<ui64>) -> tensor<ui32>
        %812 = "mhlo.convert"(%765) {name = "convert.1652"} : (tensor<ui64>) -> tensor<ui32>
        %813 = "mhlo.xor"(%811, %812) {name = "xor.1678"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %814 = "mhlo.xor"(%813, %10) {name = "xor.1679"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %815 = "mhlo.convert"(%814) {name = "convert.1686"} : (tensor<ui32>) -> tensor<ui64>
        %816 = "mhlo.multiply"(%815, %cst_7) {name = "multiply.1688"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %817 = "mhlo.convert"(%816) {name = "convert.1689"} : (tensor<ui64>) -> tensor<ui32>
        %818 = "mhlo.xor"(%809, %817) {name = "xor.1724"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %819 = "mhlo.xor"(%818, %17) {name = "xor.1725"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %820 = "mhlo.convert"(%819) {name = "convert.1737"} : (tensor<ui32>) -> tensor<ui64>
        %821 = "mhlo.multiply"(%820, %cst_6) {name = "multiply.1739"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %822 = "mhlo.shift_right_logical"(%821, %cst_8) {name = "shift-right-logical.1742"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %823 = "mhlo.convert"(%822) {name = "convert.1743"} : (tensor<ui64>) -> tensor<ui32>
        %824 = "mhlo.shift_right_logical"(%816, %cst_8) {name = "shift-right-logical.1691"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %825 = "mhlo.convert"(%824) {name = "convert.1692"} : (tensor<ui64>) -> tensor<ui32>
        %826 = "mhlo.convert"(%779) {name = "convert.1667"} : (tensor<ui64>) -> tensor<ui32>
        %827 = "mhlo.xor"(%825, %826) {name = "xor.1702"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %828 = "mhlo.xor"(%827, %18) {name = "xor.1703"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %829 = "mhlo.convert"(%828) {name = "convert.1715"} : (tensor<ui32>) -> tensor<ui64>
        %830 = "mhlo.multiply"(%829, %cst_6) {name = "multiply.1717"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %831 = "mhlo.convert"(%830) {name = "convert.1718"} : (tensor<ui64>) -> tensor<ui32>
        %832 = "mhlo.xor"(%823, %831) {name = "xor.1744"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %833 = "mhlo.xor"(%832, %7) {name = "xor.1745"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %834 = "mhlo.convert"(%833) {name = "convert.1752"} : (tensor<ui32>) -> tensor<ui64>
        %835 = "mhlo.multiply"(%834, %cst_7) {name = "multiply.1754"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %836 = "mhlo.shift_right_logical"(%835, %cst_8) {name = "shift-right-logical.1757"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %837 = "mhlo.convert"(%836) {name = "convert.1758"} : (tensor<ui64>) -> tensor<ui32>
        %838 = "mhlo.shift_right_logical"(%830, %cst_8) {name = "shift-right-logical.1720"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %839 = "mhlo.convert"(%838) {name = "convert.1721"} : (tensor<ui64>) -> tensor<ui32>
        %840 = "mhlo.convert"(%793) {name = "convert.1696"} : (tensor<ui64>) -> tensor<ui32>
        %841 = "mhlo.xor"(%839, %840) {name = "xor.1722"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %842 = "mhlo.xor"(%841, %8) {name = "xor.1723"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %843 = "mhlo.convert"(%842) {name = "convert.1730"} : (tensor<ui32>) -> tensor<ui64>
        %844 = "mhlo.multiply"(%843, %cst_7) {name = "multiply.1732"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %845 = "mhlo.convert"(%844) {name = "convert.1733"} : (tensor<ui64>) -> tensor<ui32>
        %846 = "mhlo.xor"(%837, %845) {name = "xor.1768"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %847 = "mhlo.xor"(%846, %15) {name = "xor.1769"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %848 = "mhlo.convert"(%847) {name = "convert.1781"} : (tensor<ui32>) -> tensor<ui64>
        %849 = "mhlo.multiply"(%848, %cst_6) {name = "multiply.1783"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %850 = "mhlo.shift_right_logical"(%849, %cst_8) {name = "shift-right-logical.1786"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %851 = "mhlo.convert"(%850) {name = "convert.1787"} : (tensor<ui64>) -> tensor<ui32>
        %852 = "mhlo.shift_right_logical"(%844, %cst_8) {name = "shift-right-logical.1735"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %853 = "mhlo.convert"(%852) {name = "convert.1736"} : (tensor<ui64>) -> tensor<ui32>
        %854 = "mhlo.convert"(%807) {name = "convert.1711"} : (tensor<ui64>) -> tensor<ui32>
        %855 = "mhlo.xor"(%853, %854) {name = "xor.1746"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %856 = "mhlo.xor"(%855, %16) {name = "xor.1747"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %857 = "mhlo.convert"(%856) {name = "convert.1759"} : (tensor<ui32>) -> tensor<ui64>
        %858 = "mhlo.multiply"(%857, %cst_6) {name = "multiply.1761"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %859 = "mhlo.convert"(%858) {name = "convert.1762"} : (tensor<ui64>) -> tensor<ui32>
        %860 = "mhlo.xor"(%851, %859) {name = "xor.1788"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %861 = "mhlo.xor"(%860, %5) {name = "xor.1789"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %862 = "mhlo.convert"(%861) {name = "convert.1810"} : (tensor<ui32>) -> tensor<ui64>
        %863 = "mhlo.convert"(%849) {name = "convert.1784"} : (tensor<ui64>) -> tensor<ui32>
        %864 = "mhlo.convert"(%863) {name = "convert.1811"} : (tensor<ui32>) -> tensor<ui64>
        %865 = "mhlo.shift_left"(%864, %cst_8) {name = "shift-left.1813"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %866 = "mhlo.or"(%862, %865) {name = "or.1814"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %867 = "mhlo.reshape"(%866) {name = "reshape.1815"} : (tensor<ui64>) -> tensor<1xui64>
        %868 = "mhlo.shift_left"(%23, %cst_8) {name = "shift-left.1800"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %869 = "mhlo.or"(%868, %23) {name = "or.1801"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %870 = "mhlo.reshape"(%869) {name = "reshape.1807"} : (tensor<ui64>) -> tensor<1xui64>
        %871 = "mhlo.shift_right_logical"(%858, %cst_8) {name = "shift-right-logical.1764"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %872 = "mhlo.convert"(%871) {name = "convert.1765"} : (tensor<ui64>) -> tensor<ui32>
        %873 = "mhlo.convert"(%821) {name = "convert.1740"} : (tensor<ui64>) -> tensor<ui32>
        %874 = "mhlo.xor"(%872, %873) {name = "xor.1766"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %875 = "mhlo.xor"(%874, %6) {name = "xor.1767"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %876 = "mhlo.convert"(%875) {name = "convert.1774"} : (tensor<ui32>) -> tensor<ui64>
        %877 = "mhlo.multiply"(%876, %cst_7) {name = "multiply.1776"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %878 = "mhlo.shift_right_logical"(%877, %cst_8) {name = "shift-right-logical.1779"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %879 = "mhlo.convert"(%878) {name = "convert.1780"} : (tensor<ui64>) -> tensor<ui32>
        %880 = "mhlo.convert"(%835) {name = "convert.1755"} : (tensor<ui64>) -> tensor<ui32>
        %881 = "mhlo.xor"(%879, %880) {name = "xor.1790"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %882 = "mhlo.xor"(%881, %14) {name = "xor.1791"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %883 = "mhlo.convert"(%882) {name = "convert.1802"} : (tensor<ui32>) -> tensor<ui64>
        %884 = "mhlo.convert"(%877) {name = "convert.1777"} : (tensor<ui64>) -> tensor<ui32>
        %885 = "mhlo.convert"(%884) {name = "convert.1803"} : (tensor<ui32>) -> tensor<ui64>
        %886 = "mhlo.shift_left"(%885, %cst_8) {name = "shift-left.1805"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %887 = "mhlo.or"(%883, %886) {name = "or.1806"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %888 = "mhlo.reshape"(%887) {name = "reshape.1808"} : (tensor<ui64>) -> tensor<1xui64>
        %889 = "mhlo.concatenate"(%870, %888) {dimension = 0 : i64} : (tensor<1xui64>, tensor<1xui64>) -> tensor<2xui64>
        %890 = "mhlo.concatenate"(%867, %889) {dimension = 0 : i64} : (tensor<1xui64>, tensor<2xui64>) -> tensor<3xui64>
        %891 = "mhlo.rng_bit_generator"(%890) {rng_algorithm = 2 : i32} : (tensor<3xui64>) -> tuple<tensor<3xui64>, tensor<4xui32>>
        %892 = "mhlo.slice"(%536) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xi32>) -> tensor<1x2xi32>
        %893 = "mhlo.reshape"(%892) {name = "reshape.1274"} : (tensor<1x2xi32>) -> tensor<2xi32>
        %894 = "mhlo.slice"(%893) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %895 = "mhlo.reshape"(%894) {name = "reshape.1863"} : (tensor<1xi32>) -> tensor<i32>
        %896 = "mhlo.convert"(%895) {name = "convert.1866"} : (tensor<i32>) -> tensor<ui64>
        %897 = "mhlo.slice"(%893) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
        %898 = "mhlo.reshape"(%897) {name = "reshape.1865"} : (tensor<1xi32>) -> tensor<i32>
        %899 = "mhlo.convert"(%898) {name = "convert.1867"} : (tensor<i32>) -> tensor<ui64>
        %900 = "mhlo.shift_left"(%899, %cst_8) {name = "shift-left.1869"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %901 = "mhlo.or"(%896, %900) {name = "or.1870"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %902 = "mhlo.convert"(%901) {name = "convert.1873"} : (tensor<ui64>) -> tensor<ui32>
        %903 = "mhlo.convert"(%902) {name = "convert.1876"} : (tensor<ui32>) -> tensor<ui64>
        %904 = "mhlo.convert"(%903) {name = "convert.1878"} : (tensor<ui64>) -> tensor<ui32>
        %905 = "mhlo.convert"(%904) {name = "convert.1888"} : (tensor<ui32>) -> tensor<ui64>
        %906 = "mhlo.multiply"(%905, %cst_7) {name = "multiply.1890"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %907 = "mhlo.shift_right_logical"(%906, %cst_8) {name = "shift-right-logical.1893"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %908 = "mhlo.convert"(%907) {name = "convert.1894"} : (tensor<ui64>) -> tensor<ui32>
        %909 = "mhlo.shift_right_logical"(%901, %cst_8) {name = "shift-right-logical.1874"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %910 = "mhlo.convert"(%909) {name = "convert.1875"} : (tensor<ui64>) -> tensor<ui32>
        %911 = "mhlo.convert"(%910) {name = "convert.1877"} : (tensor<ui32>) -> tensor<ui64>
        %912 = "mhlo.shift_right_logical"(%911, %cst_8) {name = "shift-right-logical.1884"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %913 = "mhlo.convert"(%912) {name = "convert.1885"} : (tensor<ui64>) -> tensor<ui32>
        %914 = "mhlo.xor"(%908, %913) {name = "xor.1904"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %915 = "mhlo.xor"(%914, %cst_5) {name = "xor.1905"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %916 = "mhlo.convert"(%915) {name = "convert.1917"} : (tensor<ui32>) -> tensor<ui64>
        %917 = "mhlo.multiply"(%916, %cst_6) {name = "multiply.1919"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %918 = "mhlo.shift_right_logical"(%917, %cst_8) {name = "shift-right-logical.1922"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %919 = "mhlo.convert"(%918) {name = "convert.1923"} : (tensor<ui64>) -> tensor<ui32>
        %920 = "mhlo.convert"(%911) {name = "convert.1882"} : (tensor<ui64>) -> tensor<ui32>
        %921 = "mhlo.convert"(%920) {name = "convert.1895"} : (tensor<ui32>) -> tensor<ui64>
        %922 = "mhlo.multiply"(%921, %cst_6) {name = "multiply.1897"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %923 = "mhlo.convert"(%922) {name = "convert.1898"} : (tensor<ui64>) -> tensor<ui32>
        %924 = "mhlo.xor"(%919, %923) {name = "xor.1924"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %925 = "mhlo.xor"(%924, %13) {name = "xor.1925"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %926 = "mhlo.convert"(%925) {name = "convert.1932"} : (tensor<ui32>) -> tensor<ui64>
        %927 = "mhlo.multiply"(%926, %cst_7) {name = "multiply.1934"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %928 = "mhlo.shift_right_logical"(%927, %cst_8) {name = "shift-right-logical.1937"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %929 = "mhlo.convert"(%928) {name = "convert.1938"} : (tensor<ui64>) -> tensor<ui32>
        %930 = "mhlo.shift_right_logical"(%922, %cst_8) {name = "shift-right-logical.1900"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %931 = "mhlo.convert"(%930) {name = "convert.1901"} : (tensor<ui64>) -> tensor<ui32>
        %932 = "mhlo.shift_right_logical"(%903, %cst_8) {name = "shift-right-logical.1880"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %933 = "mhlo.convert"(%932) {name = "convert.1881"} : (tensor<ui64>) -> tensor<ui32>
        %934 = "mhlo.xor"(%931, %933) {name = "xor.1902"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %935 = "mhlo.xor"(%934, %cst_4) {name = "xor.1903"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %936 = "mhlo.convert"(%935) {name = "convert.1910"} : (tensor<ui32>) -> tensor<ui64>
        %937 = "mhlo.multiply"(%936, %cst_7) {name = "multiply.1912"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %938 = "mhlo.convert"(%937) {name = "convert.1913"} : (tensor<ui64>) -> tensor<ui32>
        %939 = "mhlo.xor"(%929, %938) {name = "xor.1948"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %940 = "mhlo.xor"(%939, %21) {name = "xor.1949"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %941 = "mhlo.convert"(%940) {name = "convert.1961"} : (tensor<ui32>) -> tensor<ui64>
        %942 = "mhlo.multiply"(%941, %cst_6) {name = "multiply.1963"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %943 = "mhlo.shift_right_logical"(%942, %cst_8) {name = "shift-right-logical.1966"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %944 = "mhlo.convert"(%943) {name = "convert.1967"} : (tensor<ui64>) -> tensor<ui32>
        %945 = "mhlo.shift_right_logical"(%937, %cst_8) {name = "shift-right-logical.1915"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %946 = "mhlo.convert"(%945) {name = "convert.1916"} : (tensor<ui64>) -> tensor<ui32>
        %947 = "mhlo.convert"(%906) {name = "convert.1891"} : (tensor<ui64>) -> tensor<ui32>
        %948 = "mhlo.xor"(%946, %947) {name = "xor.1926"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %949 = "mhlo.xor"(%948, %22) {name = "xor.1927"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %950 = "mhlo.convert"(%949) {name = "convert.1939"} : (tensor<ui32>) -> tensor<ui64>
        %951 = "mhlo.multiply"(%950, %cst_6) {name = "multiply.1941"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %952 = "mhlo.convert"(%951) {name = "convert.1942"} : (tensor<ui64>) -> tensor<ui32>
        %953 = "mhlo.xor"(%944, %952) {name = "xor.1968"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %954 = "mhlo.xor"(%953, %11) {name = "xor.1969"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %955 = "mhlo.convert"(%954) {name = "convert.1976"} : (tensor<ui32>) -> tensor<ui64>
        %956 = "mhlo.multiply"(%955, %cst_7) {name = "multiply.1978"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %957 = "mhlo.shift_right_logical"(%956, %cst_8) {name = "shift-right-logical.1981"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %958 = "mhlo.convert"(%957) {name = "convert.1982"} : (tensor<ui64>) -> tensor<ui32>
        %959 = "mhlo.shift_right_logical"(%951, %cst_8) {name = "shift-right-logical.1944"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %960 = "mhlo.convert"(%959) {name = "convert.1945"} : (tensor<ui64>) -> tensor<ui32>
        %961 = "mhlo.convert"(%917) {name = "convert.1920"} : (tensor<ui64>) -> tensor<ui32>
        %962 = "mhlo.xor"(%960, %961) {name = "xor.1946"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %963 = "mhlo.xor"(%962, %12) {name = "xor.1947"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %964 = "mhlo.convert"(%963) {name = "convert.1954"} : (tensor<ui32>) -> tensor<ui64>
        %965 = "mhlo.multiply"(%964, %cst_7) {name = "multiply.1956"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %966 = "mhlo.convert"(%965) {name = "convert.1957"} : (tensor<ui64>) -> tensor<ui32>
        %967 = "mhlo.xor"(%958, %966) {name = "xor.1992"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %968 = "mhlo.xor"(%967, %19) {name = "xor.1993"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %969 = "mhlo.convert"(%968) {name = "convert.2005"} : (tensor<ui32>) -> tensor<ui64>
        %970 = "mhlo.multiply"(%969, %cst_6) {name = "multiply.2007"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %971 = "mhlo.shift_right_logical"(%970, %cst_8) {name = "shift-right-logical.2010"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %972 = "mhlo.convert"(%971) {name = "convert.2011"} : (tensor<ui64>) -> tensor<ui32>
        %973 = "mhlo.shift_right_logical"(%965, %cst_8) {name = "shift-right-logical.1959"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %974 = "mhlo.convert"(%973) {name = "convert.1960"} : (tensor<ui64>) -> tensor<ui32>
        %975 = "mhlo.convert"(%927) {name = "convert.1935"} : (tensor<ui64>) -> tensor<ui32>
        %976 = "mhlo.xor"(%974, %975) {name = "xor.1970"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %977 = "mhlo.xor"(%976, %20) {name = "xor.1971"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %978 = "mhlo.convert"(%977) {name = "convert.1983"} : (tensor<ui32>) -> tensor<ui64>
        %979 = "mhlo.multiply"(%978, %cst_6) {name = "multiply.1985"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %980 = "mhlo.convert"(%979) {name = "convert.1986"} : (tensor<ui64>) -> tensor<ui32>
        %981 = "mhlo.xor"(%972, %980) {name = "xor.2012"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %982 = "mhlo.xor"(%981, %9) {name = "xor.2013"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %983 = "mhlo.convert"(%982) {name = "convert.2020"} : (tensor<ui32>) -> tensor<ui64>
        %984 = "mhlo.multiply"(%983, %cst_7) {name = "multiply.2022"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %985 = "mhlo.shift_right_logical"(%984, %cst_8) {name = "shift-right-logical.2025"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %986 = "mhlo.convert"(%985) {name = "convert.2026"} : (tensor<ui64>) -> tensor<ui32>
        %987 = "mhlo.shift_right_logical"(%979, %cst_8) {name = "shift-right-logical.1988"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %988 = "mhlo.convert"(%987) {name = "convert.1989"} : (tensor<ui64>) -> tensor<ui32>
        %989 = "mhlo.convert"(%942) {name = "convert.1964"} : (tensor<ui64>) -> tensor<ui32>
        %990 = "mhlo.xor"(%988, %989) {name = "xor.1990"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %991 = "mhlo.xor"(%990, %10) {name = "xor.1991"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %992 = "mhlo.convert"(%991) {name = "convert.1998"} : (tensor<ui32>) -> tensor<ui64>
        %993 = "mhlo.multiply"(%992, %cst_7) {name = "multiply.2000"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %994 = "mhlo.convert"(%993) {name = "convert.2001"} : (tensor<ui64>) -> tensor<ui32>
        %995 = "mhlo.xor"(%986, %994) {name = "xor.2036"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %996 = "mhlo.xor"(%995, %17) {name = "xor.2037"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %997 = "mhlo.convert"(%996) {name = "convert.2049"} : (tensor<ui32>) -> tensor<ui64>
        %998 = "mhlo.multiply"(%997, %cst_6) {name = "multiply.2051"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %999 = "mhlo.shift_right_logical"(%998, %cst_8) {name = "shift-right-logical.2054"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1000 = "mhlo.convert"(%999) {name = "convert.2055"} : (tensor<ui64>) -> tensor<ui32>
        %1001 = "mhlo.shift_right_logical"(%993, %cst_8) {name = "shift-right-logical.2003"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1002 = "mhlo.convert"(%1001) {name = "convert.2004"} : (tensor<ui64>) -> tensor<ui32>
        %1003 = "mhlo.convert"(%956) {name = "convert.1979"} : (tensor<ui64>) -> tensor<ui32>
        %1004 = "mhlo.xor"(%1002, %1003) {name = "xor.2014"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1005 = "mhlo.xor"(%1004, %18) {name = "xor.2015"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1006 = "mhlo.convert"(%1005) {name = "convert.2027"} : (tensor<ui32>) -> tensor<ui64>
        %1007 = "mhlo.multiply"(%1006, %cst_6) {name = "multiply.2029"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1008 = "mhlo.convert"(%1007) {name = "convert.2030"} : (tensor<ui64>) -> tensor<ui32>
        %1009 = "mhlo.xor"(%1000, %1008) {name = "xor.2056"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1010 = "mhlo.xor"(%1009, %7) {name = "xor.2057"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1011 = "mhlo.convert"(%1010) {name = "convert.2064"} : (tensor<ui32>) -> tensor<ui64>
        %1012 = "mhlo.multiply"(%1011, %cst_7) {name = "multiply.2066"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1013 = "mhlo.shift_right_logical"(%1012, %cst_8) {name = "shift-right-logical.2069"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1014 = "mhlo.convert"(%1013) {name = "convert.2070"} : (tensor<ui64>) -> tensor<ui32>
        %1015 = "mhlo.shift_right_logical"(%1007, %cst_8) {name = "shift-right-logical.2032"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1016 = "mhlo.convert"(%1015) {name = "convert.2033"} : (tensor<ui64>) -> tensor<ui32>
        %1017 = "mhlo.convert"(%970) {name = "convert.2008"} : (tensor<ui64>) -> tensor<ui32>
        %1018 = "mhlo.xor"(%1016, %1017) {name = "xor.2034"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1019 = "mhlo.xor"(%1018, %8) {name = "xor.2035"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1020 = "mhlo.convert"(%1019) {name = "convert.2042"} : (tensor<ui32>) -> tensor<ui64>
        %1021 = "mhlo.multiply"(%1020, %cst_7) {name = "multiply.2044"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1022 = "mhlo.convert"(%1021) {name = "convert.2045"} : (tensor<ui64>) -> tensor<ui32>
        %1023 = "mhlo.xor"(%1014, %1022) {name = "xor.2080"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1024 = "mhlo.xor"(%1023, %15) {name = "xor.2081"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1025 = "mhlo.convert"(%1024) {name = "convert.2093"} : (tensor<ui32>) -> tensor<ui64>
        %1026 = "mhlo.multiply"(%1025, %cst_6) {name = "multiply.2095"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1027 = "mhlo.shift_right_logical"(%1026, %cst_8) {name = "shift-right-logical.2098"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1028 = "mhlo.convert"(%1027) {name = "convert.2099"} : (tensor<ui64>) -> tensor<ui32>
        %1029 = "mhlo.shift_right_logical"(%1021, %cst_8) {name = "shift-right-logical.2047"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1030 = "mhlo.convert"(%1029) {name = "convert.2048"} : (tensor<ui64>) -> tensor<ui32>
        %1031 = "mhlo.convert"(%984) {name = "convert.2023"} : (tensor<ui64>) -> tensor<ui32>
        %1032 = "mhlo.xor"(%1030, %1031) {name = "xor.2058"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1033 = "mhlo.xor"(%1032, %16) {name = "xor.2059"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1034 = "mhlo.convert"(%1033) {name = "convert.2071"} : (tensor<ui32>) -> tensor<ui64>
        %1035 = "mhlo.multiply"(%1034, %cst_6) {name = "multiply.2073"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1036 = "mhlo.convert"(%1035) {name = "convert.2074"} : (tensor<ui64>) -> tensor<ui32>
        %1037 = "mhlo.xor"(%1028, %1036) {name = "xor.2100"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1038 = "mhlo.xor"(%1037, %5) {name = "xor.2101"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1039 = "mhlo.convert"(%1038) {name = "convert.2122"} : (tensor<ui32>) -> tensor<ui64>
        %1040 = "mhlo.convert"(%1026) {name = "convert.2096"} : (tensor<ui64>) -> tensor<ui32>
        %1041 = "mhlo.convert"(%1040) {name = "convert.2123"} : (tensor<ui32>) -> tensor<ui64>
        %1042 = "mhlo.shift_left"(%1041, %cst_8) {name = "shift-left.2125"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1043 = "mhlo.or"(%1039, %1042) {name = "or.2126"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1044 = "mhlo.reshape"(%1043) {name = "reshape.2127"} : (tensor<ui64>) -> tensor<1xui64>
        %1045 = "mhlo.shift_left"(%23, %cst_8) {name = "shift-left.2112"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1046 = "mhlo.or"(%1045, %23) {name = "or.2113"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1047 = "mhlo.reshape"(%1046) {name = "reshape.2119"} : (tensor<ui64>) -> tensor<1xui64>
        %1048 = "mhlo.shift_right_logical"(%1035, %cst_8) {name = "shift-right-logical.2076"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1049 = "mhlo.convert"(%1048) {name = "convert.2077"} : (tensor<ui64>) -> tensor<ui32>
        %1050 = "mhlo.convert"(%998) {name = "convert.2052"} : (tensor<ui64>) -> tensor<ui32>
        %1051 = "mhlo.xor"(%1049, %1050) {name = "xor.2078"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1052 = "mhlo.xor"(%1051, %6) {name = "xor.2079"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1053 = "mhlo.convert"(%1052) {name = "convert.2086"} : (tensor<ui32>) -> tensor<ui64>
        %1054 = "mhlo.multiply"(%1053, %cst_7) {name = "multiply.2088"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1055 = "mhlo.shift_right_logical"(%1054, %cst_8) {name = "shift-right-logical.2091"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1056 = "mhlo.convert"(%1055) {name = "convert.2092"} : (tensor<ui64>) -> tensor<ui32>
        %1057 = "mhlo.convert"(%1012) {name = "convert.2067"} : (tensor<ui64>) -> tensor<ui32>
        %1058 = "mhlo.xor"(%1056, %1057) {name = "xor.2102"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1059 = "mhlo.xor"(%1058, %14) {name = "xor.2103"} : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
        %1060 = "mhlo.convert"(%1059) {name = "convert.2114"} : (tensor<ui32>) -> tensor<ui64>
        %1061 = "mhlo.convert"(%1054) {name = "convert.2089"} : (tensor<ui64>) -> tensor<ui32>
        %1062 = "mhlo.convert"(%1061) {name = "convert.2115"} : (tensor<ui32>) -> tensor<ui64>
        %1063 = "mhlo.shift_left"(%1062, %cst_8) {name = "shift-left.2117"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1064 = "mhlo.or"(%1060, %1063) {name = "or.2118"} : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
        %1065 = "mhlo.reshape"(%1064) {name = "reshape.2120"} : (tensor<ui64>) -> tensor<1xui64>
        %1066 = "mhlo.concatenate"(%1047, %1065) {dimension = 0 : i64} : (tensor<1xui64>, tensor<1xui64>) -> tensor<2xui64>
        %1067 = "mhlo.concatenate"(%1044, %1066) {dimension = 0 : i64} : (tensor<1xui64>, tensor<2xui64>) -> tensor<3xui64>
        %1068 = "mhlo.rng_bit_generator"(%1067) {rng_algorithm = 2 : i32} : (tensor<3xui64>) -> tuple<tensor<3xui64>, tensor<4xui32>>
        %1069 = "mhlo.get_tuple_element"(%891) {index = 1 : i32, name = "get-tuple-element.1818"} : (tuple<tensor<3xui64>, tensor<4xui32>>) -> tensor<4xui32>
        %1070 = "mhlo.shift_right_logical"(%1069, %29) {name = "shift-right-logical.1822"} : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
        %1071 = "mhlo.convert"(%1070) {name = "convert.1823"} : (tensor<4xui32>) -> tensor<4xf32>
        %1072 = "mhlo.multiply"(%1071, %30) {name = "multiply.1826"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1073 = "mhlo.multiply"(%1072, %31) {name = "multiply.1829"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1074 = "mhlo.add"(%1073, %35) {name = "add.1831"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1075 = "mhlo.slice"(%1074) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<2xf32>
        %1076 = "mhlo.multiply"(%1075, %24) {name = "multiply.1839"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        %1077 = "mhlo.sine"(%1076) {name = "sine.1845"} : (tensor<2xf32>) -> tensor<2xf32>
        %1078 = "mhlo.slice"(%1074) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<2xf32>
        %1079 = "mhlo.maximum"(%1078, %26) {name = "maximum.1836"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        %1080 = "mhlo.log"(%1079) {name = "log.1841"} : (tensor<2xf32>) -> tensor<2xf32>
        %1081 = "mhlo.multiply"(%1080, %25) {name = "multiply.1843"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        %1082 = "mhlo.sqrt"(%1081) {name = "sqrt.1844"} : (tensor<2xf32>) -> tensor<2xf32>
        %1083 = "mhlo.multiply"(%1077, %1082) {name = "multiply.1846"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        %1084 = "mhlo.cosine"(%1076) {name = "cosine.1847"} : (tensor<2xf32>) -> tensor<2xf32>
        %1085 = "mhlo.multiply"(%1084, %1082) {name = "multiply.1848"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        %1086 = "mhlo.concatenate"(%1083, %1085) {dimension = 0 : i64} : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
        %1087 = "mhlo.multiply"(%1086, %31) {name = "multiply.1853"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1088 = "mhlo.add"(%1087, %35) {name = "add.1856"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1089 = "mhlo.get_tuple_element"(%arg10) {index = 33 : i32, name = "get-tuple-element.650"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1090 = "mhlo.multiply"(%1089, %32) {name = "multiply.2172"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1091 = "mhlo.get_tuple_element"(%arg10) {index = 8 : i32, name = "get-tuple-element.625"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1092 = "mhlo.multiply"(%1090, %1091) {name = "multiply.2177"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1093 = "mhlo.add"(%1088, %1092) {name = "add.2178"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1094 = "mhlo.get_tuple_element"(%arg10) {index = 5 : i32, name = "get-tuple-element.622"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1095 = "mhlo.get_tuple_element"(%arg10) {index = 7 : i32, name = "get-tuple-element.624"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1096 = "mhlo.get_tuple_element"(%arg10) {index = 12 : i32, name = "get-tuple-element.629"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
        %1097 = "mhlo.get_tuple_element"(%arg10) {index = 35 : i32, name = "get-tuple-element.652"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
        %1098 = "mhlo.get_tuple_element"(%arg10) {index = 36 : i32, name = "get-tuple-element.653"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
        %1099 = "mhlo.tuple"(%cst_13, %cst_9, %cst_13, %1093, %1094, %1095, %1091, %1096, %1089, %1097, %1098) {name = "tuple.2185"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>
        %1100 = "mhlo.while"(%1099) ( {
        ^bb0(%arg11: tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>):  // no predecessors
          %1209 = "mhlo.get_tuple_element"(%arg11) {index = 2 : i32, name = "get-tuple-element.600"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
          %1210 = "mhlo.get_tuple_element"(%arg11) {index = 7 : i32, name = "get-tuple-element.605"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
          %1211 = "mhlo.compare"(%1209, %1210) {comparison_direction = "LT", name = "compare.609"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
          "mhlo.return"(%1211) : (tensor<i1>) -> ()
        },  {
        ^bb0(%arg11: tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>):  // no predecessors
          %1209 = "mhlo.get_tuple_element"(%arg11) {index = 4 : i32, name = "get-tuple-element.388"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<4xf32>
          %1210 = "mhlo.get_tuple_element"(%arg11) {index = 8 : i32, name = "get-tuple-element.392"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<4xf32>
          %1211 = "mhlo.get_tuple_element"(%arg11) {index = 3 : i32, name = "get-tuple-element.387"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<4xf32>
          %1212 = "mhlo.multiply"(%1210, %1211) {name = "multiply.455"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1213 = "mhlo.add"(%1209, %1212) {name = "add.456"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1214 = "mhlo.exponential"(%1213) {name = "exponential.486"} : (tensor<4xf32>) -> tensor<4xf32>
          %1215 = "mhlo.log"(%1214) {name = "log.493"} : (tensor<4xf32>) -> tensor<4xf32>
          %1216 = "mhlo.get_tuple_element"(%arg11) {index = 9 : i32, name = "get-tuple-element.393"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<f32>
          %1217 = "mhlo.broadcast_in_dim"(%1216) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.499"} : (tensor<f32>) -> tensor<4xf32>
          %1218 = "mhlo.divide"(%1215, %1217) {name = "divide.500"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1219 = "mhlo.get_tuple_element"(%arg11) {index = 10 : i32, name = "get-tuple-element.394"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<f32>
          %1220 = "mhlo.divide"(%1219, %1216) {name = "divide.450"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
          %1221 = "mhlo.broadcast_in_dim"(%1220) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.501"} : (tensor<f32>) -> tensor<4xf32>
          %1222 = "mhlo.subtract"(%1218, %1221) {name = "subtract.502"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1223 = "mhlo.multiply"(%1222, %28) {name = "multiply.505"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1224 = "mhlo.get_tuple_element"(%arg11) {index = 0 : i32, name = "get-tuple-element.384"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
          %1225 = "mhlo.add"(%1224, %cst_12) {name = "add.452"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
          %1226 = "mhlo.get_tuple_element"(%arg11) {index = 1 : i32, name = "get-tuple-element.385"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
          %1227 = "mhlo.get_tuple_element"(%arg11) {index = 2 : i32, name = "get-tuple-element.386"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
          %1228 = "mhlo.add"(%1227, %cst_12) {name = "add.454"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
          %1229 = "mhlo.divide"(%31, %1214) {name = "divide.489"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1230 = "mhlo.multiply"(%1229, %28) {name = "multiply.492"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1231 = "mhlo.broadcast_in_dim"(%1216) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.513"} : (tensor<f32>) -> tensor<4xf32>
          %1232 = "mhlo.divide"(%1223, %1231) {name = "divide.514"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1233 = "mhlo.divide"(%31, %1214) {name = "divide.518"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1234 = "mhlo.multiply"(%1232, %1233) {name = "multiply.519"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1235 = "mhlo.add"(%1230, %1234) {name = "add.520"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1236 = "mhlo.multiply"(%1235, %1214) {name = "multiply.521"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1237 = "mhlo.add"(%1236, %31) {name = "add.524"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1238 = "mhlo.multiply"(%1210, %1237) {name = "multiply.525"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1239 = "mhlo.add"(%1211, %1238) {name = "add.526"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1240 = "mhlo.exponential"(%1213) {name = "exponential.457"} : (tensor<4xf32>) -> tensor<4xf32>
          %1241 = "mhlo.log"(%1240) {name = "log.458"} : (tensor<4xf32>) -> tensor<4xf32>
          %1242 = "mhlo.broadcast_in_dim"(%1216) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.459"} : (tensor<f32>) -> tensor<4xf32>
          %1243 = "mhlo.divide"(%1241, %1242) {name = "divide.460"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1244 = "mhlo.divide"(%1219, %1216) {name = "divide.449"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
          %1245 = "mhlo.broadcast_in_dim"(%1244) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.461"} : (tensor<f32>) -> tensor<4xf32>
          %1246 = "mhlo.subtract"(%1243, %1245) {name = "subtract.462"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1247 = "mhlo.multiply"(%1246, %1246) {name = "multiply.463"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1248 = "mhlo.multiply"(%1247, %27) {name = "multiply.466"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1249 = "mhlo.log"(%1216) {name = "log.443"} : (tensor<f32>) -> tensor<f32>
          %1250 = "mhlo.add"(%1249, %cst_10) {name = "add.445"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
          %1251 = "mhlo.broadcast_in_dim"(%1250) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.467"} : (tensor<f32>) -> tensor<4xf32>
          %1252 = "mhlo.subtract"(%1248, %1251) {name = "subtract.468"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1253 = "mhlo.log"(%1240) {name = "log.469"} : (tensor<4xf32>) -> tensor<4xf32>
          %1254 = "mhlo.multiply"(%1253, %28) {name = "multiply.472"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1255 = "mhlo.multiply"(%1254, %31) {name = "multiply.475"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1256 = "mhlo.add"(%1252, %1255) {name = "add.476"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1257 = "mhlo.negate"(%1213) {name = "negate.477"} : (tensor<4xf32>) -> tensor<4xf32>
          %1258 = "mhlo.multiply"(%1257, %31) {name = "multiply.480"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1259 = "mhlo.negate"(%1258) {name = "negate.481"} : (tensor<4xf32>) -> tensor<4xf32>
          %1260 = "mhlo.add"(%1259, %35) {name = "add.484"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1261 = "mhlo.add"(%1256, %1260) {name = "add.485"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %1262 = "mhlo.get_tuple_element"(%arg11) {index = 7 : i32, name = "get-tuple-element.391"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
          %1263 = "mhlo.tuple"(%1225, %1226, %1228, %1239, %1213, %1261, %1237, %1262, %1210, %1216, %1219) {name = "tuple.595"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>
          "mhlo.return"(%1263) : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> ()
        }) : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>
        %1101 = "mhlo.get_tuple_element"(%1100) {index = 3 : i32, name = "get-tuple-element.2190"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<4xf32>
        %1102 = "mhlo.get_tuple_element"(%1100) {index = 4 : i32, name = "get-tuple-element.2191"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<4xf32>
        %1103 = "mhlo.get_tuple_element"(%1100) {index = 5 : i32, name = "get-tuple-element.2192"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<4xf32>
        %1104 = "mhlo.get_tuple_element"(%1100) {index = 6 : i32, name = "get-tuple-element.2193"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<f32>, tensor<f32>>) -> tensor<4xf32>
        %1105 = "mhlo.get_tuple_element"(%arg10) {index = 0 : i32, name = "get-tuple-element.617"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
        %1106 = "mhlo.add"(%1105, %cst_12) {name = "add.722"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
        %1107 = "mhlo.get_tuple_element"(%arg10) {index = 1 : i32, name = "get-tuple-element.618"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
        %1108 = "mhlo.get_tuple_element"(%arg10) {index = 2 : i32, name = "get-tuple-element.619"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
        %1109 = "mhlo.add"(%1108, %cst_12) {name = "add.724"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
        %1110 = "mhlo.slice"(%357) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xi32>) -> tensor<1x2xi32>
        %1111 = "mhlo.reshape"(%1110) {name = "reshape.999"} : (tensor<1x2xi32>) -> tensor<2xi32>
        %1112 = "mhlo.get_tuple_element"(%1068) {index = 1 : i32, name = "get-tuple-element.2130"} : (tuple<tensor<3xui64>, tensor<4xui32>>) -> tensor<4xui32>
        %1113 = "mhlo.shift_right_logical"(%1112, %29) {name = "shift-right-logical.2134"} : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
        %1114 = "mhlo.convert"(%1113) {name = "convert.2135"} : (tensor<4xui32>) -> tensor<4xf32>
        %1115 = "mhlo.multiply"(%1114, %30) {name = "multiply.2138"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1116 = "mhlo.multiply"(%1115, %31) {name = "multiply.2141"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1117 = "mhlo.add"(%1116, %35) {name = "add.2143"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1118 = "mhlo.multiply"(%1117, %31) {name = "multiply.2146"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1119 = "mhlo.add"(%1118, %35) {name = "add.2149"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1120 = "mhlo.log"(%1119) {name = "log.2150"} : (tensor<4xf32>) -> tensor<4xf32>
        %1121 = "mhlo.negate"(%1095) {name = "negate.2176"} : (tensor<4xf32>) -> tensor<4xf32>
        %1122 = "mhlo.add"(%1103, %1121) {name = "add.2224"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1123 = "mhlo.power"(%1088, %33) {name = "power.1859"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1124 = "mhlo.multiply"(%1089, %32) {name = "multiply.2175"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1125 = "mhlo.multiply"(%1124, %1104) {name = "multiply.2210"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1126 = "mhlo.subtract"(%1101, %1125) {name = "subtract.2211"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1127 = "mhlo.power"(%1126, %33) {name = "power.2214"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1128 = "mhlo.negate"(%1127) {name = "negate.2215"} : (tensor<4xf32>) -> tensor<4xf32>
        %1129 = "mhlo.add"(%1123, %1128) {name = "add.2216"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1130 = "mhlo.is_finite"(%1129) {name = "is-finite.2217"} : (tensor<4xf32>) -> tensor<4xi1>
        %1131 = "mhlo.select"(%1130, %1129, %34) {name = "select.2220"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1132 = "mhlo.multiply"(%1131, %32) {name = "multiply.2223"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1133 = "mhlo.add"(%1122, %1132) {name = "add.2225"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1134 = "mhlo.is_finite"(%1133) {name = "is-finite.2226"} : (tensor<4xf32>) -> tensor<4xi1>
        %1135 = "mhlo.select"(%1134, %1133, %34) {name = "select.2229"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1136 = "mhlo.compare"(%1120, %1135) {comparison_direction = "LT", name = "compare.2279"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %1137 = "mhlo.select"(%1136, %1102, %1094) {name = "select.2289"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1138 = "mhlo.exponential"(%1137) {name = "exponential.2290"} : (tensor<4xf32>) -> tensor<4xf32>
        %1139 = "mhlo.get_tuple_element"(%arg10) {index = 6 : i32, name = "get-tuple-element.623"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1140 = "mhlo.select"(%1136, %1132, %1139) {name = "select.2281"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1141 = "mhlo.select"(%1136, %1103, %1095) {name = "select.2283"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1142 = "mhlo.select"(%1136, %1104, %1091) {name = "select.2285"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1143 = "mhlo.get_tuple_element"(%arg10) {index = 9 : i32, name = "get-tuple-element.626"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1144 = "mhlo.select"(%1136, %1088, %1143) {name = "select.2291"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1145 = "mhlo.get_tuple_element"(%arg10) {index = 10 : i32, name = "get-tuple-element.627"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1146 = "mhlo.select"(%1136, %1126, %1145) {name = "select.2292"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1147 = "mhlo.get_tuple_element"(%arg10) {index = 25 : i32, name = "get-tuple-element.642"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
        %1148 = "mhlo.get_tuple_element"(%arg10) {index = 26 : i32, name = "get-tuple-element.643"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1149 = "mhlo.get_tuple_element"(%arg10) {index = 27 : i32, name = "get-tuple-element.644"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
        %1150 = "mhlo.get_tuple_element"(%arg10) {index = 28 : i32, name = "get-tuple-element.645"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
        %1151 = "mhlo.get_tuple_element"(%arg10) {index = 29 : i32, name = "get-tuple-element.646"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
        %1152 = "mhlo.get_tuple_element"(%arg10) {index = 30 : i32, name = "get-tuple-element.647"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1153 = "mhlo.broadcast_in_dim"(%1147) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.2152"} : (tensor<f32>) -> tensor<4xf32>
        %1154 = "mhlo.add"(%1152, %1153) {name = "add.2153"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1155 = "mhlo.is_finite"(%1135) {name = "is-finite.2230"} : (tensor<4xf32>) -> tensor<4xi1>
        %1156 = "mhlo.select"(%1155, %1135, %34) {name = "select.2233"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1157 = "mhlo.minimum"(%1156, %35) {name = "minimum.2236"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1158 = "mhlo.is_finite"(%1157) {name = "is-finite.2237"} : (tensor<4xf32>) -> tensor<4xi1>
        %1159 = "mhlo.select"(%1158, %1157, %35) {name = "select.2240"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1160 = "mhlo.subtract"(%1157, %1159) {name = "subtract.2241"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1161 = "mhlo.exponential"(%1160) {name = "exponential.2242"} : (tensor<4xf32>) -> tensor<4xf32>
        %1162 = "mhlo.log"(%1161) {name = "log.2243"} : (tensor<4xf32>) -> tensor<4xf32>
        %1163 = "mhlo.add"(%1162, %1159) {name = "add.2245"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1164 = "mhlo.subtract"(%1163, %35) {name = "subtract.2248"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1165 = "mhlo.is_finite"(%1164) {name = "is-finite.2249"} : (tensor<4xf32>) -> tensor<4xi1>
        %1166 = "mhlo.select"(%1165, %1164, %35) {name = "select.2252"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1167 = "mhlo.subtract"(%1164, %1166) {name = "subtract.2253"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1168 = "mhlo.exponential"(%1167) {name = "exponential.2254"} : (tensor<4xf32>) -> tensor<4xf32>
        %1169 = "mhlo.log"(%1168) {name = "log.2255"} : (tensor<4xf32>) -> tensor<4xf32>
        %1170 = "mhlo.add"(%1169, %1166) {name = "add.2256"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1171 = "mhlo.subtract"(%1170, %35) {name = "subtract.2259"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1172 = "mhlo.exponential"(%1171) {name = "exponential.2260"} : (tensor<4xf32>) -> tensor<4xf32>
        %1173 = "mhlo.subtract"(%1154, %1172) {name = "subtract.2261"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1174 = "mhlo.get_tuple_element"(%arg10) {index = 37 : i32, name = "get-tuple-element.654"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
        %1175 = "mhlo.get_tuple_element"(%arg10) {index = 32 : i32, name = "get-tuple-element.649"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
        %1176 = "mhlo.compare"(%1174, %1175) {comparison_direction = "LT", name = "compare.2167"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %1177 = "mhlo.broadcast_in_dim"(%1176) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.2277"} : (tensor<i1>) -> tensor<4xi1>
        %1178 = "mhlo.get_tuple_element"(%arg10) {index = 31 : i32, name = "get-tuple-element.648"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
        %1179 = "mhlo.convert"(%1175) {name = "convert.2154"} : (tensor<i32>) -> tensor<f32>
        %1180 = "mhlo.add"(%1179, %cst_11) {name = "add.2156"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
        %1181 = "mhlo.negate"(%1151) {name = "negate.2151"} : (tensor<f32>) -> tensor<f32>
        %1182 = "mhlo.power"(%1180, %1181) {name = "power.2160"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
        %1183 = "mhlo.broadcast_in_dim"(%1182) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.2269"} : (tensor<f32>) -> tensor<4xf32>
        %1184 = "mhlo.sqrt"(%1180) {name = "sqrt.2157"} : (tensor<f32>) -> tensor<f32>
        %1185 = "mhlo.broadcast_in_dim"(%1184) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.2263"} : (tensor<f32>) -> tensor<4xf32>
        %1186 = "mhlo.multiply"(%1173, %1185) {name = "multiply.2264"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1187 = "mhlo.add"(%1150, %1180) {name = "add.2158"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
        %1188 = "mhlo.multiply"(%1187, %1149) {name = "multiply.2159"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
        %1189 = "mhlo.broadcast_in_dim"(%1188) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.2265"} : (tensor<f32>) -> tensor<4xf32>
        %1190 = "mhlo.divide"(%1186, %1189) {name = "divide.2266"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1191 = "mhlo.subtract"(%1148, %1190) {name = "subtract.2267"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1192 = "mhlo.multiply"(%1183, %1191) {name = "multiply.2270"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1193 = "mhlo.subtract"(%cst_11, %1182) {name = "subtract.2162"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
        %1194 = "mhlo.broadcast_in_dim"(%1193) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.2163"} : (tensor<f32>) -> tensor<4xf32>
        %1195 = "mhlo.multiply"(%1194, %1178) {name = "multiply.2164"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1196 = "mhlo.add"(%1192, %1195) {name = "add.2271"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1197 = "mhlo.select"(%1177, %1178, %1196) {name = "select.2278"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1198 = "mhlo.add"(%1175, %cst_12) {name = "add.2169"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
        %1199 = "mhlo.compare"(%1174, %1175) {comparison_direction = "GT", name = "compare.2165"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %1200 = "mhlo.broadcast_in_dim"(%1199) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.2275"} : (tensor<i1>) -> tensor<4xi1>
        %1201 = "mhlo.exponential"(%1191) {name = "exponential.2268"} : (tensor<4xf32>) -> tensor<4xf32>
        %1202 = "mhlo.compare"(%1174, %1175) {comparison_direction = "LT", name = "compare.2166"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %1203 = "mhlo.broadcast_in_dim"(%1202) {broadcast_dimensions = dense<> : tensor<0xi64>, name = "broadcast.2273"} : (tensor<i1>) -> tensor<4xi1>
        %1204 = "mhlo.exponential"(%1196) {name = "exponential.2272"} : (tensor<4xf32>) -> tensor<4xf32>
        %1205 = "mhlo.select"(%1203, %1089, %1204) {name = "select.2274"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1206 = "mhlo.select"(%1200, %1201, %1205) {name = "select.2276"} : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        %1207 = "mhlo.get_tuple_element"(%arg10) {index = 34 : i32, name = "get-tuple-element.651"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
        %1208 = "mhlo.tuple"(%1106, %1107, %1109, %1111, %1138, %1137, %1140, %1141, %1142, %1144, %1146, %1089, %1096, %1136, %1135, %1102, %1132, %1103, %1104, %1088, %1126, %1089, %1096, %538, %359, %1147, %1148, %1149, %1150, %1151, %1173, %1197, %1198, %1206, %1207, %1097, %1098, %1174) {name = "tuple.2293"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>
        "mhlo.return"(%1208) : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> ()
      }) : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>
      %130 = "mhlo.get_tuple_element"(%129) {index = 3 : i32, name = "get-tuple-element.2404"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<2xi32>
      %131 = "mhlo.get_tuple_element"(%129) {index = 4 : i32, name = "get-tuple-element.2405"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %132 = "mhlo.get_tuple_element"(%129) {index = 5 : i32, name = "get-tuple-element.2406"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %133 = "mhlo.get_tuple_element"(%129) {index = 6 : i32, name = "get-tuple-element.2407"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %134 = "mhlo.get_tuple_element"(%129) {index = 7 : i32, name = "get-tuple-element.2408"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %135 = "mhlo.get_tuple_element"(%129) {index = 8 : i32, name = "get-tuple-element.2409"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %136 = "mhlo.get_tuple_element"(%129) {index = 9 : i32, name = "get-tuple-element.2410"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %137 = "mhlo.get_tuple_element"(%129) {index = 10 : i32, name = "get-tuple-element.2411"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %138 = "mhlo.get_tuple_element"(%129) {index = 11 : i32, name = "get-tuple-element.2412"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %139 = "mhlo.get_tuple_element"(%129) {index = 12 : i32, name = "get-tuple-element.2413"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %140 = "mhlo.get_tuple_element"(%129) {index = 13 : i32, name = "get-tuple-element.2414"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xi1>
      %141 = "mhlo.get_tuple_element"(%129) {index = 14 : i32, name = "get-tuple-element.2415"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %142 = "mhlo.get_tuple_element"(%129) {index = 15 : i32, name = "get-tuple-element.2416"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %143 = "mhlo.get_tuple_element"(%129) {index = 16 : i32, name = "get-tuple-element.2417"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %144 = "mhlo.get_tuple_element"(%129) {index = 17 : i32, name = "get-tuple-element.2418"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %145 = "mhlo.get_tuple_element"(%129) {index = 18 : i32, name = "get-tuple-element.2419"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %146 = "mhlo.get_tuple_element"(%129) {index = 19 : i32, name = "get-tuple-element.2420"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %147 = "mhlo.get_tuple_element"(%129) {index = 20 : i32, name = "get-tuple-element.2421"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %148 = "mhlo.get_tuple_element"(%129) {index = 21 : i32, name = "get-tuple-element.2422"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %149 = "mhlo.get_tuple_element"(%129) {index = 22 : i32, name = "get-tuple-element.2423"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %150 = "mhlo.get_tuple_element"(%129) {index = 23 : i32, name = "get-tuple-element.2424"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<2xi32>
      %151 = "mhlo.get_tuple_element"(%129) {index = 24 : i32, name = "get-tuple-element.2425"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<2xi32>
      %152 = "mhlo.get_tuple_element"(%129) {index = 25 : i32, name = "get-tuple-element.2426"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %153 = "mhlo.get_tuple_element"(%129) {index = 26 : i32, name = "get-tuple-element.2427"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %154 = "mhlo.get_tuple_element"(%129) {index = 27 : i32, name = "get-tuple-element.2428"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %155 = "mhlo.get_tuple_element"(%129) {index = 28 : i32, name = "get-tuple-element.2429"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %156 = "mhlo.get_tuple_element"(%129) {index = 29 : i32, name = "get-tuple-element.2430"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<f32>
      %157 = "mhlo.get_tuple_element"(%129) {index = 30 : i32, name = "get-tuple-element.2431"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %158 = "mhlo.get_tuple_element"(%129) {index = 31 : i32, name = "get-tuple-element.2432"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %159 = "mhlo.get_tuple_element"(%129) {index = 32 : i32, name = "get-tuple-element.2433"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %160 = "mhlo.get_tuple_element"(%129) {index = 33 : i32, name = "get-tuple-element.2434"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<4xf32>
      %161 = "mhlo.get_tuple_element"(%arg9) {index = 0 : i32, name = "get-tuple-element.2342"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %162 = "mhlo.add"(%161, %cst_12) {name = "add.2384"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %163 = "mhlo.get_tuple_element"(%arg9) {index = 1 : i32, name = "get-tuple-element.2343"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %164 = "mhlo.add"(%122, %cst_12) {name = "add.2386"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %165 = "mhlo.get_tuple_element"(%arg9) {index = 34 : i32, name = "get-tuple-element.2376"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tensor<i32>
      %166 = "mhlo.add"(%165, %cst_12) {name = "add.2388"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %167 = "mhlo.get_tuple_element"(%arg9) {index = 35 : i32, name = "get-tuple-element.2377"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tuple<tensor<1000x4xf32>, tensor<i32>>
      %168 = "mhlo.get_tuple_element"(%167) {index = 0 : i32, name = "get-tuple-element.2480"} : (tuple<tensor<1000x4xf32>, tensor<i32>>) -> tensor<1000x4xf32>
      %169 = "mhlo.reshape"(%131) {name = "reshape.2478"} : (tensor<4xf32>) -> tensor<1x4xf32>
      %170 = "mhlo.dynamic-update-slice"(%168, %169, %165, %cst_13) : (tensor<1000x4xf32>, tensor<1x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1000x4xf32>
      %171 = "mhlo.get_tuple_element"(%167) {index = 1 : i32, name = "get-tuple-element.2482"} : (tuple<tensor<1000x4xf32>, tensor<i32>>) -> tensor<i32>
      %172 = "mhlo.tuple"(%170, %171) {name = "tuple.2483"} : (tensor<1000x4xf32>, tensor<i32>) -> tuple<tensor<1000x4xf32>, tensor<i32>>
      %173 = "mhlo.get_tuple_element"(%arg9) {index = 36 : i32, name = "get-tuple-element.2378"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tuple<tensor<1000x4xi1>, tensor<i32>>
      %174 = "mhlo.get_tuple_element"(%173) {index = 0 : i32, name = "get-tuple-element.2486"} : (tuple<tensor<1000x4xi1>, tensor<i32>>) -> tensor<1000x4xi1>
      %175 = "mhlo.reshape"(%140) {name = "reshape.2484"} : (tensor<4xi1>) -> tensor<1x4xi1>
      %176 = "mhlo.dynamic-update-slice"(%174, %175, %165, %cst_13) : (tensor<1000x4xi1>, tensor<1x4xi1>, tensor<i32>, tensor<i32>) -> tensor<1000x4xi1>
      %177 = "mhlo.get_tuple_element"(%173) {index = 1 : i32, name = "get-tuple-element.2488"} : (tuple<tensor<1000x4xi1>, tensor<i32>>) -> tensor<i32>
      %178 = "mhlo.tuple"(%176, %177) {name = "tuple.2489"} : (tensor<1000x4xi1>, tensor<i32>) -> tuple<tensor<1000x4xi1>, tensor<i32>>
      %179 = "mhlo.tuple"(%162, %163, %164, %130, %131, %132, %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149, %150, %151, %152, %153, %154, %155, %156, %157, %158, %159, %160, %166, %172, %178, %120, %125, %126, %127) {name = "tuple.2490"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>
      "mhlo.return"(%179) : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> ()
    }) : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>
    %84 = "mhlo.get_tuple_element"(%83) {index = 35 : i32, name = "get-tuple-element.2577"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tuple<tensor<1000x4xf32>, tensor<i32>>
    %85 = "mhlo.get_tuple_element"(%83) {index = 36 : i32, name = "get-tuple-element.2578"} : (tuple<tensor<i32>, tensor<i32>, tensor<i32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<4xf32>, tensor<i32>, tuple<tensor<1000x4xf32>, tensor<i32>>, tuple<tensor<1000x4xi1>, tensor<i32>>, tuple<tensor<1000xi32>, tensor<i32>>, tensor<f32>, tensor<f32>, tensor<i32>>) -> tuple<tensor<1000x4xi1>, tensor<i32>>
    %86 = "mhlo.get_tuple_element"(%84) {index = 0 : i32, name = "get-tuple-element.2625"} : (tuple<tensor<1000x4xf32>, tensor<i32>>) -> tensor<1000x4xf32>
    %87 = "mhlo.get_tuple_element"(%85) {index = 0 : i32, name = "get-tuple-element.2626"} : (tuple<tensor<1000x4xi1>, tensor<i32>>) -> tensor<1000x4xi1>
    %88 = "mhlo.tuple"(%86, %87) {name = "tuple.2629"} : (tensor<1000x4xf32>, tensor<1000x4xi1>) -> tuple<tensor<1000x4xf32>, tensor<1000x4xi1>>
    "std.return"(%88) : (tuple<tensor<1000x4xf32>, tensor<1000x4xi1>>) -> ()
  }) {sym_name = "main", type = (tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<i32>) -> tuple<tensor<1000x4xf32>, tensor<1000x4xi1>>} : () -> ()
  "module_terminator"() : () -> ()
}) : () -> ()
