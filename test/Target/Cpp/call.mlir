// RUN: emitc-translate -mlir-to-c %s | FileCheck %s -check-prefix=C-DEFAULT
// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: emitc-translate -mlir-to-c-with-variable-declarations-at-top %s | FileCheck %s -check-prefix=C-FWDDECL
// RUN: emitc-translate -mlir-to-cpp-with-variable-declarations-at-top %s | FileCheck %s -check-prefix=CPP-FWDDECL

func @emitc_call() {
  %0 = emitc.call "func_a" () : () -> i32
  %1 = emitc.call "func_b" () : () -> i32
  return
}
// C-DEFAULT: void emitc_call() {
// C-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = func_a();
// C-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = func_b();

// CPP-DEFAULT: void emitc_call() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = func_a();
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = func_b();

// C-FWDDECL: void emitc_call() {
// C-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// C-FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// C-FWDDECL-NEXT: [[V0:]] = func_a();
// C-FWDDECL-NEXT: [[V1:]] = func_b();

// CPP-FWDDECL: void emitc_call() {
// CPP-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V0:]] = func_a();
// CPP-FWDDECL-NEXT: [[V1:]] = func_b();
