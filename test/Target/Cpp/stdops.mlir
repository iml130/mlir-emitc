// RUN: emitc-translate -mlir-to-c %s | FileCheck %s -check-prefix=C-DEFAULT
// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: emitc-translate -mlir-to-c-forward-declared %s | FileCheck %s -check-prefix=C-FWDDECL
// RUN: emitc-translate -mlir-to-cpp-forward-declared %s | FileCheck %s -check-prefix=CPP-FWDDECL

func @std_constant() {
  %c0 = constant 0 : i32
  %c1 = constant 2 : index
  %c2 = constant 2.0 : f32
  return
}
// C-DEFAULT: void std_constant() {
// C-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = 0;
// C-DEFAULT-NEXT: size_t [[V1:[^ ]*]] = 2;
// C-DEFAULT-NEXT: float [[V2:[^ ]*]] = (float)2.000000000e+00;

// CPP-DEFAULT: void std_constant() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]]{0};
// CPP-DEFAULT-NEXT: size_t [[V1:[^ ]*]]{2};
// CPP-DEFAULT-NEXT: float [[V2:[^ ]*]]{(float)2.000000000e+00};

// C-FWDDECL: void std_constant() {
// C-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// C-FWDDECL-NEXT: size_t [[V1:[^ ]*]];
// C-FWDDECL-NEXT: float [[V2:[^ ]*]];
// C-FWDDECL-NEXT: [[V0]] = 0;
// C-FWDDECL-NEXT: [[V1]] = 2;
// C-FWDDECL-NEXT: [[V2]] = (float)2.000000000e+00;

// CPP-FWDDECL: void std_constant() {
// CPP-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// CPP-FWDDECL-NEXT: size_t [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: float [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V0]] = 0;
// CPP-FWDDECL-NEXT: [[V1]] = 2;
// CPP-FWDDECL-NEXT: [[V2]] = (float)2.000000000e+00;

func @std_call() {
  %0 = call @one_result () : () -> i32
  %1 = call @one_result () : () -> i32
  return
}
// C-DEFAULT: void std_call() {
// C-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = one_result();
// C-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = one_result();

// CPP-DEFAULT: void std_call() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = one_result();
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = one_result();

// C-FWDDECL: void std_call() {
// C-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// C-FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// C-FWDDECL-NEXT: [[V0]] = one_result();
// C-FWDDECL-NEXT: [[V1]] = one_result();

// CPP-FWDDECL: void std_call() {
// CPP-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V0]] = one_result();
// CPP-FWDDECL-NEXT: [[V1]] = one_result();


func @one_result() -> i32 {
  %0 = constant 0 : i32
  return %0 : i32
}
// C-DEFAULT: int32_t one_result() {
// C-DEFAULT-NEXT: int32_t [[V0:[^ ]*]] = 0;
// C-DEFAULT-NEXT: return [[V0]];

// CPP-DEFAULT: int32_t one_result() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]]{0};
// CPP-DEFAULT-NEXT: return [[V0]];

// C-FWDDECL: int32_t one_result() {
// C-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// C-FWDDECL-NEXT: [[V0]] = 0;
// C-FWDDECL-NEXT: return [[V0]];

// CPP-FWDDECL: int32_t one_result() {
// CPP-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V0]] = 0;
// CPP-FWDDECL-NEXT: return [[V0]];


func @single_return_statement(%arg0 : i32) -> i32 {
  return %arg0 : i32
}
// C-DEFAULT: int32_t single_return_statement(int32_t [[V0:[^ ]*]]) {
// C-DEFAULT-NEXT: return [[V0]];

// CPP-DEFAULT: int32_t single_return_statement(int32_t [[V0:[^ ]*]]) {
// CPP-DEFAULT-NEXT: return [[V0]];

// C-FWDDECL: int32_t single_return_statement(int32_t [[V0:[^ ]*]]) {
// C-FWDDECL-NEXT: return [[V0]];

// CPP-FWDDECL: int32_t single_return_statement(int32_t [[V0:[^ ]*]]) {
// CPP-FWDDECL-NEXT: return [[V0]];
