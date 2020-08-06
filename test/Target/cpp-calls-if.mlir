// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s --dump-input-on-failure

// CHECK: // Forward declare functions.
// CHECK: void std_if(bool, float);
// CHECK: void std_if_else(bool, float);

func @std_if(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
    %0 = addf %arg1, %arg1 : f32
  }
  return
}

func @std_if_else(%arg0: i1, %arg1: f32) {
  emitc.if %arg0 {
    %0 = addf %arg1, %arg1 : f32
  } else {
    %1 = addf %arg1, %arg1 : f32
  }
  return
}
