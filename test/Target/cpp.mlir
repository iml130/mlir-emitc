// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s

// CHECK: void emitc_call() {
func @emitc_call() {
  // CHECK-NEXT: int32_t [[V0:[^ ]*]] = one_result();
  %0 = emitc.call "one_result" () : () -> i32
  // CHECK-NEXT: int32_t [[V1:[^ ]*]];
  // CHECK-NEXT: int32_t [[V2:[^ ]*]];
  // CHECK-NEXT: std::tie([[V1]], [[V2]]) = two_results();
  %1:2 = emitc.call "two_results" () : () -> (i32, i32)
  return
}
