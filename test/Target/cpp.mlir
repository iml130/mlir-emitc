// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s

// CHECK: // Forward declare functions.
// CHECK: void std_constant();
// CHECK: void std_call();
// CHECK: void emitc_constant();
// CHECK: void emitc_call();
// CHECK: void emitc_if();
// CHECK: void emitc_for();

// CHECK: void std_constant()
func @std_constant() {
  // CHECK-NEXT: [[V0:[^ ]*]]{0};
  %c0 = constant 0 : i32
  // CHECK-NEXT: [[V1:[^ ]*]]{2};
  %c1 = constant 2 : index
  // CHECK-NEXT: [[V2:[^ ]*]]{(float)2.000000000e+00};
  %c2 = constant 2.0 : f32
  // CHECK-NEXT: [[V3:[^ ]*]]{1, 1};
  %c3 = constant dense<1> : tensor<2xi32>
  return
}

func @std_call() {
  // TODO(simon-camp): add test
  return
}

func @emitc_constant() {
  // TODO(simon-camp): add test
  return
}

func @emitc_call() {
  // TODO(simon-camp): add test
  return
}

func @emitc_if() {
  // TODO(simon-camp): add test
  return
}

func @emitc_for() {
  // TODO(simon-camp): add test
  return
}
