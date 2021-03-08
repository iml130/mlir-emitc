// RUN: emitc-opt %s -split-input-file -verify-diagnostics

func @index_args_out_of_range_1() {
    emitc.call "test" () {args = [0 : index]} : () -> () // expected-error {{'emitc.call' op index argument is out of range}}
    return
}

// -----

func @index_args_out_of_range_2(%arg : i32) {
    emitc.call "test" (%arg, %arg) {args = [2 : index]} : (i32, i32) -> () // expected-error {{'emitc.call' op index argument is out of range}}
    return
}

// -----

func @empty_callee() {
    emitc.call "" () : () -> () // expected-error {{'emitc.call' op callee must not be empty}}
    return
}
