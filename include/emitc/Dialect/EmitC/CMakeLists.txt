add_mlir_dialect(EmitC emitc)
add_mlir_doc(EmitC EmitC Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EmitC)
add_public_tablegen_target(MLIREmitCPassIncGen)
