if(${EMITC_ENABLE_HLO})

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}
  )

set(LLVM_TARGET_DEFINITIONS MHLOToStd.td)
mlir_tablegen(MHLOToStd.inc -gen-rewriters)
add_public_tablegen_target(MLIRMHLOToStdIncGen)

add_mlir_library(MLIREmitCConversion
  MHLOToStd.cpp

  DEPENDS
  MLIRLmhloPassIncGen
  MLIRMhloPassIncGen
  MLIRMHLOToStdIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MhloRegisterDialects
  )
endif()
