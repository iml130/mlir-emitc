add_mlir_dialect(EmitC emitc)
add_mlir_doc(EmitC -gen-dialect-doc EmitC Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EmitC)
add_public_tablegen_target(MLIREmitCPassIncGen)
