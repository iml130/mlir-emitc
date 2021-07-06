// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: emitc-translate -mlir-to-cpp-with-variable-declarations-at-top %s | FileCheck %s -check-prefix=CPP-FWDDECL

func @emitc_call() {
  %0 = emitc.call "func_a" () : () -> i32
  %1 = emitc.call "func_b" () : () -> i32
  return
}
// CPP-DEFAULT: void emitc_call() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = func_a();
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = func_b();

// CPP-FWDDECL: void emitc_call() {
// CPP-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V0:]] = func_a();
// CPP-FWDDECL-NEXT: [[V1:]] = func_b();
