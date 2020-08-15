if(${EMITC_ENABLE_HLO})
add_mlir_library(MLIREmitCTransforms
  ConvertToScf.cpp

  DEPENDS
  MLIRLmhloPassIncGen
  MLIRMhloPassIncGen

  LINK_LIBS PUBLIC
  MLIRLoopLikeInterface
  MLIRSCF
  MLIRPass
  MLIRTransformUtils
  MhloRegisterDialects
  )
endif()
