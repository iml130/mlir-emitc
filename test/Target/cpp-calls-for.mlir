// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s --dump-input-on-failure

// CHECK: // Forward declare functions.
// CHECK: void test_for(size_t, size_t, size_t);

func @test_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  emitc.for %i0 = %arg0 to %arg1 step %arg2 {
  }
  return
}

func @std_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = emitc.call "foo::constant"() {args = [0 : i32]} : () -> (i32)
  %result = emitc.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (i32) {
    %sn = emitc.call "foo::add"(%si) {args = [0 : index]} : (i32) -> (i32)
    emitc.yield %sn : i32
  }
  return
}
