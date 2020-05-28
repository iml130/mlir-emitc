// RUN: mlir-opt -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @f(%{{.*}}: i32, %{{.*}}: !emitc.int32_t) -> i1 {
func @f(%arg0: i32, %f: !emitc<"int32_t">) -> i1 {
  %1 = "emitc.call"() {callee = "blah"} : () -> i64
  emitc.call "foo" (%1) {args = [
    0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index
  ]} : (i64) -> ()
  %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
  return %2#1 : i1
}
