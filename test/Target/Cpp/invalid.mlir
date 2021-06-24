// RUN: emitc-translate -split-input-file -mlir-to-cpp -verify-diagnostics %s
// RUN: emitc-translate -split-input-file -mlir-to-c -verify-diagnostics %s

// expected-error@+1 {{'func' op with multiple blocks needs forward declared variables}}
func @multiple_blocks() {
^bb1:
    br ^bb2
^bb2:
    return
}

// -----

func @unsupported_std_op(%arg0: f64) -> f64 {
  // expected-error@+1 {{'std.absf' op unable to find printer for op}}
  %0 = absf %arg0 : f64
  return %0 : f64
}

// -----

// expected-error@+1 {{cannot emit integer type 'i80'}}
func @unsupported_integer_type(%arg0 : i80) {
  return
}

// -----

// expected-error@+1 {{cannot emit float type 'f80'}}
func @unsupported_float_type(%arg0 : f80) {
  return
}

// -----

// expected-error@+1 {{cannot emit type 'memref<100xf32>'}}
func @memref_type(%arg0 : memref<100xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit type 'vector<100xf32>'}}
func @vector_type(%arg0 : vector<100xf32>) {
  return
}
