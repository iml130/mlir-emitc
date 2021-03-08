// RUN: emitc-translate -mlir-to-c %s | FileCheck %s

// CHECK: // Forward declare functions.
// CHECK: void emitc_constant();

// CHECK: void emitc_constant()
func @emitc_constant() {
  // CHECK: int32_t [[V1:[^ ]*]] = 42;
  %1 = "emitc.const"(){value = 42 : i32} : () -> i32
  // CHECK: int32_t* [[V2:[^ ]*]] = NULL;
  %2 = "emitc.const"(){value = "NULL" : !emitc.opaque<"..">} : () -> !emitc.opaque<"int32_t*">
  return
}
