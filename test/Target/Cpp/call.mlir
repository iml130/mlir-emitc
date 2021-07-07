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


func @emitc_call_two_results() {
  %0 = constant 0 : index
  %1:2 = emitc.call "two_results" () : () -> (i32, i32)
  return
}
// CPP-DEFAULT: void emitc_call_two_results() {
// CPP-DEFAULT-NEXT: size_t [[V1:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: std::tie([[V2]], [[V3]]) = two_results();

// CPP-FWDDECL: void emitc_call_two_results() {
// CPP-FWDDECL-NEXT: size_t [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V3:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V1]] = 0;
// CPP-FWDDECL-NEXT: std::tie([[V2]], [[V3]]) = two_results();
