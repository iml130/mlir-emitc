add_mlir_dialect_library(MLIREmitC
  EmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${EMITC_MAIN_INCLUDE_DIR}/emitc/Dialect/EmitC

  DEPENDS
  MLIREmitCIncGen
  MLIREmitCAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
