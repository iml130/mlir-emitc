// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s --dump-input-on-failure

// CHECK: // Forward declare functions.
// CHECK: void test_for(size_t, size_t, size_t);

func @test_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  emitc.for %i0 = %arg0 to %arg1 step %arg2 {
  }
  return
}
