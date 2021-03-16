// RUN: emitc-translate -mlir-to-c %s | FileCheck %s -check-prefix=DEFAULT
// RUN: emitc-translate -mlir-to-c -forward-declare-variables %s | FileCheck %s -check-prefix=FWDDECL

func @emitc_call() {
  %0 = emitc.call "func_a" () : () -> i32
  %1 = emitc.call "func_b" () : () -> i32
  return
}
// DEFAULT: void emitc_call() {
// DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = func_a();
// DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = func_b();

// FWDDECL: void emitc_call() {
// FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// FWDDECL-NEXT: [[V0:]] = func_a();
// FWDDECL-NEXT: [[V1:]] = func_b();
