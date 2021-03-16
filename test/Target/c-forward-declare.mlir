// RUN: emitc-translate -mlir-to-c -forward-declare-variables %s | FileCheck %s

// CHECK: void emitc_call() {
func @emitc_call() {
  // CHECK-NEXT: int32_t [[V0:[^ ]*]];
  // CHECK-NEXT: int32_t [[V1:[^ ]*]];
  // CHECK-NEXT: [[V0:]] = one_result();
  // CHECK-NEXT: [[V1:]] = one_result();
  %0 = emitc.call "one_result" () : () -> i32
  %1 = emitc.call "one_result" () : () -> i32
  return
}
