// RUN: emitc-translate -mlir-to-c -forward-declare-variables %s | FileCheck %s

// CHECK: void std_constant() {
func @std_constant() {
  // CHECK-NEXT: int32_t [[V0:[^ ]*]];
  // CHECK-NEXT: size_t [[V1:[^ ]*]];
  // CHECK-NEXT: float [[V2:[^ ]*]];

  // CHECK-NEXT: [[V0]] = 0;
  %c0 = constant 0 : i32
  // CHECK-NEXT: [[V1]] = 2;
  %c1 = constant 2 : index
  // CHECK-NEXT: [[V2]] = (float)2.000000000e+00;
  %c2 = constant 2.0 : f32
  return
}

// CHECK: void std_call() {
func @std_call() {
  // CHECK-NEXT: int32_t [[V0:[^ ]*]];
  // CHECK-NEXT: int32_t [[V1:[^ ]*]];
  // CHECK-NEXT: [[V0:]] = one_result();
  // CHECK-NEXT: [[V1:]] = one_result();
  %0 = call @one_result () : () -> i32
  %1 = call @one_result () : () -> i32
  return
}

func @emitc_constant() {
  // TODO(simon-camp): add test
  return
}

// CHECK: void emitc_call() {
func @emitc_call() {
  // CHECK-NEXT: int32_t [[V0:[^ ]*]];
  // CHECK-NEXT: int32_t [[V1:[^ ]*]];
  // CHECK-NEXT: [[V0:]] = one_result();
  // CHECK-NEXT: [[V1:]] = one_result();
  %0 = emitc.call "one_result" () : () -> i32
  %1 = emitc.call "one_result" () : () -> i32
  return
}

func @one_result() -> i32 {
  %0 = constant 0 : i32
  return %0 : i32
}
