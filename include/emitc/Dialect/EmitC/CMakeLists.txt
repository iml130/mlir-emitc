add_mlir_dialect(EmitC emitc)
add_mlir_doc(EmitC EmitC Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS EmitC.td)
mlir_tablegen(EmitCAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(EmitCAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIREmitCAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EmitC)
add_public_tablegen_target(MLIREmitCPassIncGen)
