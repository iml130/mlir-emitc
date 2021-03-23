// RUN: emitc-translate -mlir-to-c -forward-declare-variables %s | FileCheck %s -check-prefix=C-FWDDECL
// RUN: emitc-translate -mlir-to-cpp -forward-declare-variables %s | FileCheck %s -check-prefix=CPP-FWDDECL

// simple(10, true)  -> 20
// simple(10, false) -> 30
func @simple(i64, i1) -> i64 {
^bb0(%a: i64, %cond: i1):
  cond_br %cond, ^bb1, ^bb2
^bb1:
  br ^bb3(%a: i64)
^bb2:
  %b = emitc.call "add"(%a, %a) : (i64, i64) -> i64
  br ^bb3(%b: i64)
^bb3(%c: i64):
  br ^bb4(%c, %a : i64, i64)
^bb4(%d : i64, %e : i64):
  %0 = emitc.call "add"(%d, %e) : (i64, i64) -> i64
  return %0 : i64
}
// C-FWDDECL: int64_t simple(int64_t [[A:[^ ]*]], bool [[COND:[^ ]*]]) {
  // C-FWDDECL-NEXT: int64_t [[B:[^ ]*]];
  // C-FWDDECL-NEXT: int64_t [[V0:[^ ]*]];
  // C-FWDDECL-NEXT: int64_t [[C:[^ ]*]];
  // C-FWDDECL-NEXT: int64_t [[D:[^ ]*]];
  // C-FWDDECL-NEXT: int64_t [[E:[^ ]*]];
  // C-FWDDECL-NEXT: [[BB0:[^ ]*]]:
  // C-FWDDECL-NEXT: if ([[COND]]) {
  // C-FWDDECL-NEXT: goto [[BB1:[^ ]*]];
  // C-FWDDECL-NEXT: } else {
  // C-FWDDECL-NEXT: goto [[BB2:[^ ]*]];
  // C-FWDDECL-NEXT: }
  // C-FWDDECL-NEXT: [[BB1]]:
  // C-FWDDECL-NEXT: [[C]] = [[A]];
  // C-FWDDECL-NEXT: goto [[BB3:[^ ]*]];
  // C-FWDDECL-NEXT: [[BB2]]:
  // C-FWDDECL-NEXT: [[B]] = add([[A]], [[A]]);
  // C-FWDDECL-NEXT: [[C]] = [[B]];
  // C-FWDDECL-NEXT: goto [[BB3]];
  // C-FWDDECL-NEXT: [[BB3]]:
  // C-FWDDECL-NEXT: [[D]] = [[C]];
  // C-FWDDECL-NEXT: [[E]] = [[A]];
  // C-FWDDECL-NEXT: goto [[BB4:[^ ]*]];
  // C-FWDDECL-NEXT: [[BB4]]:
  // C-FWDDECL-NEXT: [[V0]] = add([[D]], [[E]]);
  // C-FWDDECL-NEXT: return [[V0]];

  // CPP-FWDDECL: int64_t simple(int64_t [[A:[^ ]*]], bool [[COND:[^ ]*]]) {
    // CPP-FWDDECL-NEXT: int64_t [[B:[^ ]*]];
    // CPP-FWDDECL-NEXT: int64_t [[V0:[^ ]*]];
    // CPP-FWDDECL-NEXT: int64_t [[C:[^ ]*]];
    // CPP-FWDDECL-NEXT: int64_t [[D:[^ ]*]];
    // CPP-FWDDECL-NEXT: int64_t [[E:[^ ]*]];
    // CPP-FWDDECL-NEXT: [[BB0:[^ ]*]]:
    // CPP-FWDDECL-NEXT: if ([[COND]]) {
    // CPP-FWDDECL-NEXT: goto [[BB1:[^ ]*]];
    // CPP-FWDDECL-NEXT: } else {
    // CPP-FWDDECL-NEXT: goto [[BB2:[^ ]*]];
    // CPP-FWDDECL-NEXT: }
    // CPP-FWDDECL-NEXT: [[BB1]]:
    // CPP-FWDDECL-NEXT: [[C]] = [[A]];
    // CPP-FWDDECL-NEXT: goto [[BB3:[^ ]*]];
    // CPP-FWDDECL-NEXT: [[BB2]]:
    // CPP-FWDDECL-NEXT: [[B]] = add([[A]], [[A]]);
    // CPP-FWDDECL-NEXT: [[C]] = [[B]];
    // CPP-FWDDECL-NEXT: goto [[BB3]];
    // CPP-FWDDECL-NEXT: [[BB3]]:
    // CPP-FWDDECL-NEXT: [[D]] = [[C]];
    // CPP-FWDDECL-NEXT: [[E]] = [[A]];
    // CPP-FWDDECL-NEXT: goto [[BB4:[^ ]*]];
    // CPP-FWDDECL-NEXT: [[BB4]]:
    // CPP-FWDDECL-NEXT: [[V0]] = add([[D]], [[E]]);
    // CPP-FWDDECL-NEXT: return [[V0]];
