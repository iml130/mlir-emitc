// This file contains tests for emitc call ops which are only supported if cpp code is emitted.

// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: emitc-translate -mlir-to-cpp -forward-declare-variables %s | FileCheck %s -check-prefix=CPP-FWDDECL

func @emitc_call() {
  %0 = constant 0 : index
  %1:2 = emitc.call "two_results" () : () -> (i32, i32)
  return
}
// CPP-DEFAULT: void emitc_call() {
// CPP-DEFAULT-NEXT: size_t [[V1:[^ ]*]]{0};
// CPP-DEFAULT-NEXT: int32_t [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: std::tie([[V2]], [[V3]]) = two_results();

// CPP-FWDDECL: void emitc_call() {
// CPP-FWDDECL-NEXT: size_t [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V3:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V1]] = 0;
// CPP-FWDDECL-NEXT: std::tie([[V2]], [[V3]]) = two_results();
