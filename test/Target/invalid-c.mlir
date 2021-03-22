// RUN: emitc-translate -split-input-file -mlir-to-c -verify-diagnostics %s

func @func_template() {
  // expected-error @+1 {{'emitc.call' op template arguments are not supported if emitting C}}
  %0 = emitc.call "func_template" () {template_args = [i32]} : () -> i32
  return
}

// -----
