// RUN: emitc-translate -split-input-file -mlir-to-cpp -verify-diagnostics %s

// expected-error@+1 {{cannot emit tensor type with non static shape}}
func @non_static_shape(%arg0 : tensor<?xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit unranked tensor type}}
func @unranked_tensor(%arg0 : tensor<*xf32>) {
  return
}
