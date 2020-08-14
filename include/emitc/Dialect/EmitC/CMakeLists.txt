add_mlir_dialect(EmitC emitc EmitC)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EmitC)
add_public_tablegen_target(MLIREmitCPassIncGen)
