// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s

// CHECK: void std_constant() {
func @std_constant() {
  // CHECK-NEXT: int32_t [[V0:[^ ]*]]{0};
  %c0 = constant 0 : i32
  // CHECK-NEXT: size_t [[V1:[^ ]*]]{2};
  %c1 = constant 2 : index
  // CHECK-NEXT: float [[V2:[^ ]*]]{(float)2.000000000e+00};
  %c2 = constant 2.0 : f32
  // CHECK-NEXT: Tensor<int32_t, 2> [[V3:[^ ]*]]{1, 2};
  %c3 = constant dense<[1, 2]> : tensor<2xi32>
  return
}

// CHECK: void std_call() {
func @std_call() {
  // CHECK-NEXT: int32_t [[V0:[^ ]*]] = one_result();
  %0 = call @one_result () : () -> i32
  // CHECK-NEXT: int32_t [[V1:[^ ]*]];
  // CHECK-NEXT: int32_t [[V2:[^ ]*]];
  // CHECK-NEXT: std::tie([[V1]], [[V2]]) = two_results();
  %1:2 = call @two_results () : () -> (i32, i32)
  return
}

func @emitc_constant() {
  // TODO(simon-camp): add test
  return
}

// CHECK: void emitc_call() {
func @emitc_call() {
  // CHECK-NEXT: int32_t [[V0:[^ ]*]] = one_result();
  %0 = emitc.call "one_result" () : () -> i32
  // CHECK-NEXT: int32_t [[V1:[^ ]*]];
  // CHECK-NEXT: int32_t [[V2:[^ ]*]];
  // CHECK-NEXT: std::tie([[V1]], [[V2]]) = two_results();
  %1:2 = emitc.call "two_results" () : () -> (i32, i32)
  return
}

func @one_result() -> i32 {
  %0 = constant 0 : i32
  return %0 : i32
}

func @two_results() -> (i32, i32) {
  %0 = constant 0 : i32
  return %0, %0 : i32, i32
}
