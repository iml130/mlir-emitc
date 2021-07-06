// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: emitc-translate -mlir-to-cpp-with-variable-declarations-at-top %s | FileCheck %s -check-prefix=CPP-FWDDECL


func @emitc_constant() {
  %c0 = "emitc.constant"(){value = #emitc.opaque<""> : i32} : () -> i32
  %c1 = "emitc.constant"(){value = 42 : i32} : () -> i32
  %c2 = "emitc.constant"(){value = #emitc.opaque<""> : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
  %c3 = "emitc.constant"(){value = #emitc.opaque<"NULL"> : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
  return
}
// CPP-DEFAULT: void emitc_constant() {
// CPP-DEFAULT-NEXT: int32_t [[V0:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]] = 42;
// CPP-DEFAULT-NEXT: int32_t* [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t* [[V3:[^ ]*]] = NULL;

// CPP-FWDDECL: void emitc_constant() {
// CPP-FWDDECL-NEXT: int32_t [[V0:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t* [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t* [[V3:[^ ]*]];
// CPP-FWDDECL-NEXT: ;
// CPP-FWDDECL-NEXT: [[V1]] = 42;
// CPP-FWDDECL-NEXT: ;
// CPP-FWDDECL-NEXT: [[V3]] = NULL;
