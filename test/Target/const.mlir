// RUN: emitc-translate -mlir-to-c %s | FileCheck %s -check-prefix=C-DEFAULT
// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: emitc-translate -mlir-to-c-forward-declared %s | FileCheck %s -check-prefix=C-FWDDECL
// RUN: emitc-translate -mlir-to-cpp-forward-declared %s | FileCheck %s -check-prefix=CPP-FWDDECL


func @emitc_constant() {
  %c0 = "emitc.const"(){value = #emitc.opaque<""> : i32} : () -> i32
  %c1 = "emitc.const"(){value = 42 : i32} : () -> i32
  %c2 = "emitc.const"(){value = #emitc.opaque<""> : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
  %c3 = "emitc.const"(){value = #emitc.opaque<"NULL"> : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
  return
}
// C-DEFAULT: void emitc_constant() {
// C-DEFAULT-NEXT: int32_t [[V0:[^ ]*]];
// C-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = 42;
// C-DEFAULT-NEXT: int32_t* [[V2:[^ ]*]];
// C-DEFAULT-NEXT: int32_t* [[V3:[^ ]*]] = NULL;

// CPP-DEFAULT: void emitc_constant() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]]{42};
// CPP-DEFAULT-NEXT: int32_t* [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t* [[V3:[^ ]*]]{NULL};

// C-FWDDECL: void emitc_constant() {
// C-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// C-FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// C-FWDDECL-NEXT: int32_t* [[V2:[^ ]*]];
// C-FWDDECL-NEXT: int32_t* [[V3:[^ ]*]];
// C-FWDDECL-NEXT: ;
// C-FWDDECL-NEXT: [[V1]] = 42;
// C-FWDDECL-NEXT: ;
// C-FWDDECL-NEXT: [[V3]] = NULL;

// CPP-FWDDECL: void emitc_constant() {
// CPP-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t* [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t* [[V3:[^ ]*]];
// CPP-FWDDECL-NEXT: ;
// CPP-FWDDECL-NEXT: [[V1]] = 42;
// CPP-FWDDECL-NEXT: ;
// CPP-FWDDECL-NEXT: [[V3]] = NULL;
