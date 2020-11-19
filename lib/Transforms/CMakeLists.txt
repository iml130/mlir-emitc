if(${EMITC_ENABLE_HLO})

include_directories(BEFORE
  ${mlir-hlo_BINARY_DIR}/lib/Dialect/mhlo/transforms
  )

add_mlir_library(MLIREmitCTransforms
  MHLOToStd.cpp

  DEPENDS
  MLIRLmhloPassIncGen
  MLIRMhloPassIncGen
  MLIRMhloLegalizeToStandardIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MhloRegisterDialects
  )
endif()
