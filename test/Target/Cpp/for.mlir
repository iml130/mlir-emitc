// RUN: emitc-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT
// RUN: emitc-translate -mlir-to-cpp-with-variable-declarations-at-top %s | FileCheck %s -check-prefix=CPP-FWDDECL

func @test_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    %0 = emitc.call "f"() : () -> i32
  }
  return
}
// CPP-DEFAULT: void test_for(size_t [[START:[^ ]*]], size_t [[STOP:[^ ]*]], size_t [[STEP:[^ ]*]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-DEFAULT-NEXT: int32_t [[V4:[^ ]*]] = f();
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-EMPTY:
// CPP-DEFAULT-NEXT: return;

// CPP-FWDDECL: void test_for(size_t [[START:[^ ]*]], size_t [[STOP:[^ ]*]], size_t [[STEP:[^ ]*]]) {
// CPP-FWDDECL-NEXT: int32_t [[V4:[^ ]*]];
// CPP-FWDDECL-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-FWDDECL-NEXT: [[V4]] = f();
// CPP-FWDDECL-NEXT: }
// CPP-FWDDECL-EMPTY:
// CPP-FWDDECL-NEXT: return;

func @test_for_yield() {
  %start = constant 0 : index
  %stop = constant 10 : index
  %step = constant 1 : index

  %s0 = constant 0 : i32
  %p0 = constant 1.0 : f32
  
  %result:2 = scf.for %iter = %start to %stop step %step iter_args(%si = %s0, %pi = %p0) -> (i32, f32) {
    %sn = emitc.call "add"(%si, %iter) : (i32, index) -> i32
    %pn = emitc.call "mul"(%pi, %iter) : (f32, index) -> f32
    scf.yield %sn, %pn : i32, f32
  }

  return
}
// CPP-DEFAULT: void test_for_yield() {
// CPP-DEFAULT-NEXT: size_t [[START:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: size_t [[STOP:[^ ]*]] = 10;
// CPP-DEFAULT-NEXT: size_t [[STEP:[^ ]*]] = 1;
// CPP-DEFAULT-NEXT: int32_t [[S0:[^ ]*]] = 0;
// CPP-DEFAULT-NEXT: float [[P0:[^ ]*]] = (float)1.000000000e+00;
// CPP-DEFAULT-NEXT: int32_t [[SE:[^ ]*]];
// CPP-DEFAULT-NEXT: float [[PE:[^ ]*]];
// CPP-DEFAULT-NEXT: int32_t [[SI:[^ ]*]] = [[S0]];
// CPP-DEFAULT-NEXT: float [[PI:[^ ]*]] = [[P0]];
// CPP-DEFAULT-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-DEFAULT-NEXT: int32_t [[SN:[^ ]*]] = add([[SI]], [[ITER]]);
// CPP-DEFAULT-NEXT: float [[PN:[^ ]*]] = mul([[PI]], [[ITER]]);
// CPP-DEFAULT-NEXT: [[SI]] = [[SN]];
// CPP-DEFAULT-NEXT: [[PI]] = [[PN]];
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: [[SE]] = [[SI]];
// CPP-DEFAULT-NEXT: [[PE]] = [[PI]];
// CPP-DEFAULT-EMPTY:
// CPP-DEFAULT-NEXT: return;

// CPP-FWDDECL: void test_for_yield() {
// CPP-FWDDECL-NEXT: size_t [[START:[^ ]*]];
// CPP-FWDDECL-NEXT: size_t [[STOP:[^ ]*]];
// CPP-FWDDECL-NEXT: size_t [[STEP:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[S0:[^ ]*]];
// CPP-FWDDECL-NEXT: float [[P0:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[SE:[^ ]*]];
// CPP-FWDDECL-NEXT: float [[PE:[^ ]*]];
// CPP-FWDDECL-NEXT: int32_t [[SN:[^ ]*]];
// CPP-FWDDECL-NEXT: float [[PN:[^ ]*]];
// CPP-FWDDECL-NEXT: [[START]] = 0;
// CPP-FWDDECL-NEXT: [[STOP]] = 10;
// CPP-FWDDECL-NEXT: [[STEP]] = 1;
// CPP-FWDDECL-NEXT: [[S0]] = 0;
// CPP-FWDDECL-NEXT: [[P0]] = (float)1.000000000e+00;
// CPP-FWDDECL-NEXT: int32_t [[SI:[^ ]*]] = [[S0]];
// CPP-FWDDECL-NEXT: float [[PI:[^ ]*]] = [[P0]];
// CPP-FWDDECL-NEXT: for (size_t [[ITER:[^ ]*]] = [[START]]; [[ITER]] < [[STOP]]; [[ITER]] += [[STEP]]) {
// CPP-FWDDECL-NEXT: [[SN]] = add([[SI]], [[ITER]]);
// CPP-FWDDECL-NEXT: [[PN]] = mul([[PI]], [[ITER]]);
// CPP-FWDDECL-NEXT: [[SI]] = [[SN]];
// CPP-FWDDECL-NEXT: [[PI]] = [[PN]];
// CPP-FWDDECL-NEXT: }
// CPP-FWDDECL-NEXT: [[SE]] = [[SI]];
// CPP-FWDDECL-NEXT: [[PE]] = [[PI]];
// CPP-FWDDECL-EMPTY:
// CPP-FWDDECL-NEXT: return;
