add_mlir_library(MLIRMHLOToEmitC
  MHLOToEmitC.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
