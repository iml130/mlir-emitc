add_mlir_library(MLIRTensorToEmitC
  TensorToEmitC.cpp

  DEPENDS
  MLIREmitCDialect
  MLIREmitCConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
