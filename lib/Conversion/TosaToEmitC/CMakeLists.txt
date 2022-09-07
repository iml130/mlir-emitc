add_mlir_library(MLIRTosaToEmitC
  TosaToEmitC.cpp

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
