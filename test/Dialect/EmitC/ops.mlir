// RUN: emitc-opt -allow-unregistered-dialect -verify-diagnostics %s | FileCheck %s

"emitc.include" (){include = "<test.h>"} : () -> ()
emitc.include "<test.h>"

// CHECK-LABEL: func @f(%{{.*}}: i32, %{{.*}}: !custom.int32_t) -> i1 {
func @f(%arg0: i32, %f: !custom<"int32_t">) -> i1 {
  %1 = "emitc.call"() {callee = "blah"} : () -> i64
  emitc.call "foo" (%1) {args = [
    0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index
  ]} : (i64) -> ()
  %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
  return %2#1 : i1
}

func @c(%arg0: i32) {
  %1 = "emitc.const"(){value = 42 : i32} : () -> i32
  return
}

func @a(%arg0: i32, %arg1: i32) {
  %1 = "emitc.apply"(%arg0) {applicableOperator = "&"} : (i32) -> !emitc.opaque<"int32_t*">
  %2 = emitc.apply "&"(%arg1) : (i32) -> !emitc.opaque<"int32_t*">
  return
}
