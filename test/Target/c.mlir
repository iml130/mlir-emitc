// RUN: emitc-translate -mlir-to-c %s | FileCheck %s -check-prefix=DEFAULT
// RUN: emitc-translate -mlir-to-c -forward-declare-variables %s | FileCheck %s -check-prefix=FWDDECL

func @std_constant() {
  %c0 = constant 0 : i32
  %c1 = constant 2 : index
  %c2 = constant 2.0 : f32
  return
}
// DEFAULT: void std_constant() {
// DEFAULT-NEXT: [[V0:[^ ]*]] = 0;
// DEFAULT-NEXT: [[V1:[^ ]*]] = 2;
// DEFAULT-NEXT: [[V2:[^ ]*]] = (float)2.000000000e+00;

// FWDDECL: void std_constant() {
// FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// FWDDECL-NEXT: size_t [[V1:[^ ]*]];
// FWDDECL-NEXT: float [[V2:[^ ]*]];
// FWDDECL-NEXT: [[V0]] = 0;
// FWDDECL-NEXT: [[V1]] = 2;
// FWDDECL-NEXT: [[V2]] = (float)2.000000000e+00;

func @std_call() {
  %0 = call @one_result () : () -> i32
  %1 = call @one_result () : () -> i32
  return
}
// DEFAULT: void std_call() {
// DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = one_result();
// DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = one_result();

// FWDDECL: void std_call() {
// FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// FWDDECL-NEXT: [[V0:]] = one_result();
// FWDDECL-NEXT: [[V1:]] = one_result();

func @emitc_constant() {
  // TODO(simon-camp): add test
  return
}

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

func @one_result() -> i32 {
  %0 = constant 0 : i32
  return %0 : i32
}
