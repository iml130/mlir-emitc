// RUN: emitc-translate -split-input-file -mlir-to-c -verify-diagnostics %s

func @func_template() {
  // expected-error @+1 {{'emitc.call' op template arguments are not supported if emitting C}}
  %0 = emitc.call "func_template" () {template_args = [i32]} : () -> i32
  return
}

// -----

func @func_tuple() {
  // expected-error @+1 {{cannot emit tuple type if emitting C}}
  %cst = "emitc.constant"(){value = tuple<ui64, ui32>} : () -> i32
  return
}

// -----

func @func_tensor() {
  // expected-error @+1 {{cannot emit tensor type if emitting C}}
  %cst = "emitc.constant"(){value = dense<1> : tensor<24xi32>} : () -> tensor<24xi32>
  return
}

// -----
