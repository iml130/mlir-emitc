add_mlir_dialect(EmitC emitc)
add_mlir_doc(EmitC EmitC Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS EmitCAttrDefs.td)
mlir_tablegen(EmitCAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(EmitCAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIREmitCAttrDefsIncGen)
