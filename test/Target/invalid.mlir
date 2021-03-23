// RUN: emitc-translate -mlir-to-cpp -split-input-file -verify-diagnostics %s
// RUN: emitc-translate -mlir-to-c -split-input-file -verify-diagnostics %s

// expected-error@+1 {{'func' op with multiple blocks needs forward declared variables}}
func @multiple_blocks() {
^bb1:
    br ^bb2
^bb2:
    return
}
