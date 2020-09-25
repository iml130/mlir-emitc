// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s 

// CHECK: // Forward declare functions.
// CHECK: void test_if(bool, float);
// CHECK: void test_if_else(bool, float);
// CHECK: void test_if_yield(bool, float);

// void test_if(bool v1, float v2)
func @test_if(%arg0: i1, %arg1: f32) {
  // CHECK: if (v1) {
  emitc.if %arg0 {
     %0 = emitc.call "foo::constant"() {args = [dense<[0, 1]> : tensor<2xi32>]} : () -> (i32)
  }
  return
}

// void test_if_else(bool v1, float v2)
func @test_if_else(%arg0: i1, %arg1: f32) {
  // CHECK: if (v1) {
  emitc.if %arg0 {
    %0 = emitc.call "foo::constant"() {args = [dense<[0, 1]> : tensor<2xi32>]} : () -> (i32)
  // CHECK } else {
  } else {
    %0 = emitc.call "foo::constant"() {args = [dense<[1, 1]> : tensor<2xi32>]} : () -> (i32)
  }
  return
}

// void test_if_yield(bool v1, float v2)
func @test_if_yield(%arg0: i1, %arg1: f32) {
  // CHECK: int32_t v3;
  // CHECK: int32_t v4;
  // CHECK: if (v1) {
  %x, %y = emitc.if %arg0 -> (i32, i32) {
    %0 = emitc.call "foo::constant"() {args = [dense<[0, 0]> : tensor<2xi32>]} : () -> (i32)
    %1 = emitc.call "foo::constant"() {args = [dense<[0, 1]> : tensor<2xi32>]} : () -> (i32)
    // CHECK: v3 = v5;
    // CHECK: v4 = v6;
    emitc.yield %0, %1 : i32, i32
  // CHECK } else {
  } else {
    %0 = emitc.call "foo::constant"() {args = [dense<[1, 0]> : tensor<2xi32>]} : () -> (i32)
    %1 = emitc.call "foo::constant"() {args = [dense<[1, 1]> : tensor<2xi32>]} : () -> (i32)
    // CHECK: v3 = v7;
    // CHECK: v4 = v8;
    emitc.yield %0, %1 : i32, i32
  }
  return
}
