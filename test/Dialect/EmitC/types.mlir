// RUN: emitc-opt -allow-unregistered-dialect -verify-diagnostics %s | FileCheck %s
// check parser
// RUN: emitc-opt -allow-unregistered-dialect -verify-diagnostics %s | emitc-opt -allow-unregistered-dialect -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @opaque_types() {
func @opaque_types() {
  // CHECK-NEXT: !emitc.opaque<"int">
  emitc.call ""() {args = [!emitc<"opaque<\"int\">">]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"byte">
  emitc.call ""() {args = [!emitc<"opaque<\"byte\">">]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"unsigned">
  emitc.call ""() {args = [!emitc<"opaque<\"unsigned\">">]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"status_t">
  emitc.call ""() {args = [!emitc<"opaque<\"status_t\">">]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"std::vector<std::string>">
  emitc.call ""() {args = [!emitc.opaque<"std::vector<std::string>">]} : () -> ()
  return
}
