// RUN: emitc-translate -mlir-to-c %s | FileCheck %s

// CHECK: // Forward declare functions.
// CHECK: void emitc_constant();

// CHECK: void emitc_constant()
func @emitc_constant() {
  // CHECK: int32_t [[V0:[^ ]*]];
  %c0 = "emitc.const"(){value = "" : i32} : () -> i32
  // CHECK: int32_t [[V1:[^ ]*]] = 42;
  %c1 = "emitc.const"(){value = 42 : i32} : () -> i32
  // CHECK: int32_t* [[V2:[^ ]*]];
  %c2 = "emitc.const"(){value = "" : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
  // CHECK: int32_t* [[V3:[^ ]*]] = NULL;
  %c3 = "emitc.const"(){value = "NULL" : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
  return
}
