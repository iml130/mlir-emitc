// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s 

// CHECK: // Forward declare functions.
// CHECK: void test_for(size_t, size_t, size_t);
// CHECK: void test_for_yield(size_t, size_t, size_t);

// CHECK: void test_for(size_t v1, size_t v2, size_t v3) {
func @test_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: for (size_t v4=v1; v4<v2; v4=v4+v3) {
  emitc.for %i0 = %arg0 to %arg1 step %arg2 {
  }
  return
}

// CHECK: void test_for_yield(size_t v1, size_t v2, size_t v3) {
func @test_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = emitc.call "foo::constant"() {args = [0 : i32]} : () -> (i32)
  // CHECK: for (size_t v7=v1; v7<v2; v7=v7+v3) {
  %result = emitc.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (i32) {
    %sn = emitc.call "foo::add"(%si) {args = [0 : index]} : (i32) -> (i32)
    emitc.yield %sn : i32
  }
  return
}
