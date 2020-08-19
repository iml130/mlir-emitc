// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s 

// CHECK: // Forward declare functions.
// CHECK: void test_foo_print();
// CHECK: int32_t test_single_return(int32_t);
// CHECK: std::tuple<int32_t, int32_t> test_multiple_return();

// CHECK: void test_foo_print()
func @test_foo_print() {
  // CHECK: [[V1:[^ ]*]] = foo::constant({0, 1});
  %0 = emitc.call "foo::constant"() {args = [dense<[0, 1]> : tensor<2xi32>]} : () -> (i32)
  // CHECK: [[V2:[^ ]*]] = foo::op_and_attr({0, 1}, [[V1]]);
  %1 = emitc.call "foo::op_and_attr"(%0) {args = [dense<[0, 1]> : tensor<2xi32>, 0 : index]} : (i32) -> (i32)
  // CHECK: [[V3:[^ ]*]] = foo::op_and_attr([[V2]], {0, 1});
  %2 = emitc.call "foo::op_and_attr"(%1) {args = [0 : index, dense<[0, 1]> : tensor<2xi32>]} : (i32) -> (i32)
  // CHECK: foo::print([[V3]]);
  emitc.call "foo::print"(%2): (i32) -> ()
  return
}

// CHECK: int32_t test_single_return(int32_t [[V2:.*]])
func @test_single_return(%arg0 : i32) -> i32 {
  // CHECK: return [[V2]]
  return %arg0 : i32
}

// CHECK: std::tuple<int32_t, int32_t> test_multiple_return()
func @test_multiple_return() -> (i32, i32) {
  // CHECK: std::tie([[V3:.*]], [[V4:.*]]) = foo::blah();
  %0:2 = emitc.call "foo::blah"() : () -> (i32, i32)
  // CHECK: [[V5:[^ ]*]] = test_single_return([[V3]]);
  %1 = call @test_single_return(%0#0) : (i32) -> i32
  // CHECK: return std::make_tuple([[V5]], [[V4]]);
  return %1, %0#1 : i32, i32
}

// CHECK: test_float
func @test_float() {
  // CHECK: foo::constant({(float)0.0e+00, (float)1.000000000e+00})
  %0 = emitc.call "foo::constant"() {args = [dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>]} : () -> f32
  return
}

// CHECK: test_uint
func @test_uint() {
  // CHECK: uint32_t
  %0 = emitc.call "foo::constant"() {args = [dense<[0, 1]> : tensor<2xui32>]} : () -> ui32
  // CHECK: uint64_t
  %1 = emitc.call "foo::constant"() {args = [dense<[0, 1]> : tensor<2xui64>]} : () -> ui64
  return
}

// CHECK: int64_t test_plus_int(int64_t [[V1]])
func @test_plus_int(%arg0 : i64) -> i64 {
  // CHECK: mhlo::add([[V1]], [[V1]])
  %0 = emitc.call "mhlo::add"(%arg0, %arg0) {args = [0 : index, 1 : index]} : (i64, i64) -> i64
  return %0 : i64
}
