// RUN: emitc-opt %s -split-input-file -verify-diagnostics

func @index_args_out_of_range_1() {
    // expected-error @+1 {{'emitc.call' op index argument is out of range}}
    emitc.call "test" () {args = [0 : index]} : () -> ()
    return
}

// -----

func @index_args_out_of_range_2(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op index argument is out of range}}
    emitc.call "test" (%arg, %arg) {args = [2 : index]} : (i32, i32) -> ()
    return
}

// -----

func @empty_callee() {
    // expected-error @+1 {{'emitc.call' op callee must not be empty}}
    emitc.call "" () : () -> ()
    return
}

// -----

func @nonetype_arg(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op array argument has no type}}
    emitc.call "nonetype_arg"(%arg) {args = [0 : index, [0, 1, 2]]} : (i32) -> i32
    return
}

// -----

func @nonetype_template_arg(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op array template argument has no type}}
    emitc.call "nonetype_template_arg"(%arg) {template_args = [[0, 1, 2]]} : (i32) -> i32
    return
}

// -----

func @float_template_argument(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op float literal as template argument is invalid}}
    emitc.call "float_template_argument"(%arg) {template_args = [0.5 : f32]} : (i32) -> i32
    return
}

// -----

func @dense_template_argument(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op dense float elements as template argument are invalid}}
    emitc.call "dense_template_argument"(%arg) {template_args = [dense<[1.0, 1.0]> : tensor<2xf32>]} : (i32) -> i32
    return
}
