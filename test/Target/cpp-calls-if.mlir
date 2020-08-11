// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s --dump-input-on-failure

// CHECK: // Forward declare functions.
// CHECK: void std_if(bool, float);
// CHECK: void std_if_else(bool, float);

func @std_if(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
     %0 = emitc.call "foo::constant"() {args = [dense<[0, 1]> : tensor<2xi32>]} : () -> (i32)
  }
  return
}

func @std_if_else(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
    %0 = emitc.call "foo::constant"() {args = [dense<[0, 1]> : tensor<2xi32>]} : () -> (i32)
  } else {
    %0 = emitc.call "foo::constant"() {args = [dense<[1, 1]> : tensor<2xi32>]} : () -> (i32)
  }
  return
}

func @std_if_yield(%arg0: i1, %arg1: f32)
{
  %x, %y = emitc.if %arg0 -> (i32, i32) {
    %0 = emitc.call "foo::constant"() {args = [dense<[0, 0]> : tensor<2xi32>]} : () -> (i32)
    %1 = emitc.call "foo::constant"() {args = [dense<[0, 1]> : tensor<2xi32>]} : () -> (i32)
    emitc.yield %0, %1 : i32, i32
  } else {
    %0 = emitc.call "foo::constant"() {args = [dense<[1, 0]> : tensor<2xi32>]} : () -> (i32)
    %1 = emitc.call "foo::constant"() {args = [dense<[1, 1]> : tensor<2xi32>]} : () -> (i32)
    emitc.yield %0, %1 : i32, i32
  }
  return
}
