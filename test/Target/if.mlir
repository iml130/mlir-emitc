// RUN: emitc-translate -mlir-to-c %s | FileCheck %s -check-prefix=C-DEFAULT
// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: emitc-translate -mlir-to-c-forward-declared %s | FileCheck %s -check-prefix=C-FWDDECL
// RUN: emitc-translate -mlir-to-cpp-forward-declared %s | FileCheck %s -check-prefix=CPP-FWDDECL

func @test_if(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
     %0 = emitc.call "func_const"(%arg1) : (f32) -> i32
  }
  return
}
// C-DEFAULT: void test_if(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// C-DEFAULT-NEXT: if ([[V0]]) {
// C-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = func_const([[V1]]);
// C-DEFAULT-NEXT: ;
// C-DEFAULT-NEXT: }
// C-DEFAULT-EMPTY:
// C-DEFAULT-NEXT: return;

// CPP-DEFAULT: void test_if(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: if ([[V0]]) {
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = func_const([[V1]]);
// CPP-DEFAULT-NEXT: ;
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-EMPTY:
// CPP-DEFAULT-NEXT: return;

// C-FWDDECL: void test_if(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// C-FWDDECL-NEXT: int32_t [[V2:[^ ]*]];
// C-FWDDECL-NEXT: if ([[V0]]) {
// C-FWDDECL-NEXT: [[V2]] = func_const([[V1]]);
// C-FWDDECL-NEXT: ;
// C-FWDDECL-NEXT: }
// C-FWDDECL-EMPTY:
// C-FWDDECL-NEXT: return;

// CPP-FWDDECL: void test_if(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-FWDDECL-NEXT: int32_t [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: if ([[V0]]) {
// CPP-FWDDECL-NEXT: [[V2]] = func_const([[V1]]);
// CPP-FWDDECL-NEXT: ;
// CPP-FWDDECL-NEXT: }
// CPP-FWDDECL-EMPTY:
// CPP-FWDDECL-NEXT: return;


func @test_if_else(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
    %0 = emitc.call "func_true"(%arg1) : (f32) -> i32
  } else {
    %0 = emitc.call "func_false"(%arg1) : (f32) -> i32
  }
  return
}
// C-DEFAULT: void test_if_else(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// C-DEFAULT-NEXT: if ([[V0]]) {
// C-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = func_true([[V1]]);
// C-DEFAULT-NEXT: ;
// C-DEFAULT-NEXT: }
// C-DEFAULT-NEXT: else {
// C-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = func_false([[V1]]);
// C-DEFAULT-NEXT: ;
// C-DEFAULT-NEXT: }
// C-DEFAULT-EMPTY:
// C-DEFAULT-NEXT: return;

// CPP-DEFAULT: void test_if_else(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: if ([[V0]]) {
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]] = func_true([[V1]]);
// CPP-DEFAULT-NEXT: ;
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: else {
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]] = func_false([[V1]]);
// CPP-DEFAULT-NEXT: ;
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-EMPTY:
// CPP-DEFAULT-NEXT: return;

// C-FWDDECL: void test_if_else(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// C-FWDDECL-NEXT: int32_t [[V2:[^ ]*]];
// C-FWDDECL-NEXT: int32_t [[V3:[^ ]*]];
// C-FWDDECL-NEXT: if ([[V0]]) {
// C-FWDDECL-NEXT: [[V2]] = func_true([[V1]]);
// C-FWDDECL-NEXT: ;
// C-FWDDECL-NEXT: }
// C-FWDDECL-NEXT: else {
// C-FWDDECL-NEXT: [[V3]] = func_false([[V1]]);
// C-FWDDECL-NEXT: ;
// C-FWDDECL-NEXT: }
// C-FWDDECL-EMPTY:
// C-FWDDECL-NEXT: return;

// CPP-FWDDECL: void test_if_else(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-FWDDECL-NEXT: int32_t [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V3:[^ ]*]];
// CPP-FWDDECL-NEXT: if ([[V0]]) {
// CPP-FWDDECL-NEXT: [[V2]] = func_true([[V1]]);
// CPP-FWDDECL-NEXT: ;
// CPP-FWDDECL-NEXT: }
// CPP-FWDDECL-NEXT: else {
// CPP-FWDDECL-NEXT: [[V3]] = func_false([[V1]]);
// CPP-FWDDECL-NEXT: ;
// CPP-FWDDECL-NEXT: }
// CPP-FWDDECL-EMPTY:
// CPP-FWDDECL-NEXT: return;


func @test_if_yield(%arg0: i1, %arg1: f32) {
  %0 = constant 0 : i8
  %x, %y = scf.if %arg0 -> (i32, f64) {
    %1 = emitc.call "func_true_1"(%arg1) : (f32) -> i32
    %2 = emitc.call "func_true_2"(%arg1) : (f32) -> f64
    scf.yield %1, %2 : i32, f64
  } else {
    %1 = emitc.call "func_false_1"(%arg1) : (f32) -> i32
    %2 = emitc.call "func_false_2"(%arg1) : (f32) -> f64
    scf.yield %1, %2 : i32, f64
  }
  return
}
// C-DEFAULT: void test_if_yield(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// C-DEFAULT-NEXT: int8_t [[V2:[^ ]*]] = 0;
// C-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// C-DEFAULT-NEXT: double [[V4:[^ ]*]];
// C-DEFAULT-NEXT: if ([[V0]]) {
// C-DEFAULT-NEXT: int32_t [[V5:[^ ]*]] = func_true_1([[V1]]);
// C-DEFAULT-NEXT: double [[V6:[^ ]*]] = func_true_2([[V1]]);
// C-DEFAULT-NEXT: [[V3]] = [[V5]];
// C-DEFAULT-NEXT: [[V4]] = [[V6]];
// C-DEFAULT-NEXT: }
// C-DEFAULT-NEXT: else {
// C-DEFAULT-NEXT: int32_t [[V7:[^ ]*]] = func_false_1([[V1]]);
// C-DEFAULT-NEXT: double [[V8:[^ ]*]] = func_false_2([[V1]]);
// C-DEFAULT-NEXT: [[V3]] = [[V7]];
// C-DEFAULT-NEXT: [[V4]] = [[V8]];
// C-DEFAULT-NEXT: }
// C-DEFAULT-EMPTY:
// C-DEFAULT-NEXT: return;

// CPP-DEFAULT: void test_if_yield(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-DEFAULT-NEXT: int8_t [[V2:[^ ]*]]{0};
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: double [[V4:[^ ]*]];
// CPP-DEFAULT-NEXT: if ([[V0]]) {
// CPP-DEFAULT-NEXT: int32_t [[V5:[^ ]*]] = func_true_1([[V1]]);
// CPP-DEFAULT-NEXT: double [[V6:[^ ]*]] = func_true_2([[V1]]);
// CPP-DEFAULT-NEXT: [[V3]] = [[V5]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V6]];
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: else {
// CPP-DEFAULT-NEXT: int32_t [[V7:[^ ]*]] = func_false_1([[V1]]);
// CPP-DEFAULT-NEXT: double [[V8:[^ ]*]] = func_false_2([[V1]]);
// CPP-DEFAULT-NEXT: [[V3]] = [[V7]];
// CPP-DEFAULT-NEXT: [[V4]] = [[V8]];
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-EMPTY:
// CPP-DEFAULT-NEXT: return;

// C-FWDDECL: void test_if_yield(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// C-FWDDECL-NEXT: int8_t [[V2:[^ ]*]];
// C-FWDDECL-NEXT: int32_t [[V3:[^ ]*]];
// C-FWDDECL-NEXT: double [[V4:[^ ]*]];
// C-FWDDECL-NEXT: int32_t [[V5:[^ ]*]];
// C-FWDDECL-NEXT: double [[V6:[^ ]*]];
// C-FWDDECL-NEXT: int32_t [[V7:[^ ]*]];
// C-FWDDECL-NEXT: double [[V8:[^ ]*]];
// C-FWDDECL-NEXT: [[V2]] = 0;
// C-FWDDECL-NEXT: if ([[V0]]) {
// C-FWDDECL-NEXT: [[V5]] = func_true_1([[V1]]);
// C-FWDDECL-NEXT: [[V6]] = func_true_2([[V1]]);
// C-FWDDECL-NEXT: [[V3]] = [[V5]];
// C-FWDDECL-NEXT: [[V4]] = [[V6]];
// C-FWDDECL-NEXT: }
// C-FWDDECL-NEXT: else {
// C-FWDDECL-NEXT: [[V7]] = func_false_1([[V1]]);
// C-FWDDECL-NEXT: [[V8]] = func_false_2([[V1]]);
// C-FWDDECL-NEXT: [[V3]] = [[V7]];
// C-FWDDECL-NEXT: [[V4]] = [[V8]];
// C-FWDDECL-NEXT: }
// C-FWDDECL-EMPTY:
// C-FWDDECL-NEXT: return;

// CPP-FWDDECL: void test_if_yield(bool [[V0:[^ ]*]], float [[V1:[^ ]*]]) {
// CPP-FWDDECL-NEXT: int8_t [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V3:[^ ]*]];
// CPP-FWDDECL-NEXT: double [[V4:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V5:[^ ]*]];
// CPP-FWDDECL-NEXT: double [[V6:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V7:[^ ]*]];
// CPP-FWDDECL-NEXT: double [[V8:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V2]] = 0;
// CPP-FWDDECL-NEXT: if ([[V0]]) {
// CPP-FWDDECL-NEXT: [[V5]] = func_true_1([[V1]]);
// CPP-FWDDECL-NEXT: [[V6]] = func_true_2([[V1]]);
// CPP-FWDDECL-NEXT: [[V3]] = [[V5]];
// CPP-FWDDECL-NEXT: [[V4]] = [[V6]];
// CPP-FWDDECL-NEXT: }
// CPP-FWDDECL-NEXT: else {
// CPP-FWDDECL-NEXT: [[V7]] = func_false_1([[V1]]);
// CPP-FWDDECL-NEXT: [[V8]] = func_false_2([[V1]]);
// CPP-FWDDECL-NEXT: [[V3]] = [[V7]];
// CPP-FWDDECL-NEXT: [[V4]] = [[V8]];
// CPP-FWDDECL-NEXT: }
// CPP-FWDDECL-EMPTY:
// CPP-FWDDECL-NEXT: return;
