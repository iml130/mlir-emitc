add_mlir_dialect_library(MLIREmitC
  IR/EmitCDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/EmitC

  DEPENDS
  MLIREmitCIncGen
  )
target_link_libraries(MLIREmitC
  PUBLIC
  MLIRIR
  MLIRSideEffects
  LLVMSupport
  )
