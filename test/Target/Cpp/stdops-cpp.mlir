// This file contains tests for std ops which are only supported if cpp code is emitted.

// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: emitc-translate -mlir-to-cpp-with-variable-declarations-at-top %s | FileCheck %s -check-prefix=CPP-FWDDECL

func @std_constant() {
  %c0 = constant dense<0> : tensor<i32>
  %c1 = constant dense<[0, 1]> : tensor<2xindex>
  %c2 = constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
  return
}
// CPP-DEFAULT: void std_constant() {
// CPP-DEFAULT-NEXT: Tensor<int32_t> [[V0:[^ ]*]]{0};
// CPP-DEFAULT-NEXT: Tensor<size_t, 2> [[V1:[^ ]*]]{0, 1};
// CPP-DEFAULT-NEXT: Tensor<float, 2, 2> [[V2:[^ ]*]]{(float)0.0e+00, (float)1.000000000e+00, (float)2.000000000e+00, (float)3.000000000e+00};

// CPP-FWDDECL: void std_constant() {
// CPP-FWDDECL-NEXT: Tensor<int32_t> [[V0:[^ ]*]];
// CPP-FWDDECL-NEXT: Tensor<size_t, 2> [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: Tensor<float, 2, 2> [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V0]] = {0};
// CPP-FWDDECL-NEXT: [[V1]] = {0, 1};
// CPP-FWDDECL-NEXT: [[V2]] = {(float)0.0e+00, (float)1.000000000e+00, (float)2.000000000e+00, (float)3.000000000e+00};

func @std_call() {
  %c = constant 0 : i8
  %0:2 = call @two_results () : () -> (i32, f32)
  %1:2 = call @two_results () : () -> (i32, f32)
  return
}
// CPP-DEFAULT: void std_call() {
// CPP-DEFAULT-NEXT: int8_t  [[V0:[^ ]*]]{0};
// CPP-DEFAULT-NEXT: int32_t [[V1:[^ ]*]];
// CPP-DEFAULT-NEXT: float [[V2:[^ ]*]];
// CPP-DEFAULT-NEXT: std::tie([[V1]], [[V2]]) = two_results();
// CPP-DEFAULT-NEXT: int32_t [[V3:[^ ]*]];
// CPP-DEFAULT-NEXT: float [[V4:[^ ]*]];
// CPP-DEFAULT-NEXT: std::tie([[V3]], [[V4]]) = two_results();

// CPP-FWDDECL: void std_call() {
// CPP-FWDDECL-NEXT: int8_t [[V0:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V1:[^ ]*]];
// CPP-FWDDECL-NEXT: float [[V2:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[V3:[^ ]*]];
// CPP-FWDDECL-NEXT: float [[V4:[^ ]*]];
// CPP-FWDDECL-NEXT: [[V0]] = 0;
// CPP-FWDDECL-NEXT: std::tie([[V1]], [[V2]]) = two_results();
// CPP-FWDDECL-NEXT: std::tie([[V3]], [[V4]]) = two_results();


func @two_results() -> (i32, f32) {
  %0 = constant 0 : i32
  %1 = constant 1.0 : f32
  return %0, %1 : i32, f32
}
// CPP-DEFAULT: std::tuple<int32_t, float> two_results() {
// CPP-DEFAULT: int32_t [[V0:[^ ]*]]{0};
// CPP-DEFAULT: float [[V1:[^ ]*]]{(float)1.000000000e+00};
// CPP-DEFAULT: return std::make_tuple([[V0]], [[V1]]);

// CPP-FWDDECL: std::tuple<int32_t, float> two_results() {
// CPP-FWDDECL: int32_t [[V0:[^ ]*]];
// CPP-FWDDECL: float [[V1:[^ ]*]];
// CPP-FWDDECL: [[V0]] = 0;
// CPP-FWDDECL: [[V1]] = (float)1.000000000e+00;
// CPP-FWDDECL: return std::make_tuple([[V0]], [[V1]]);
