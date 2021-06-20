add_mlir_library(MLIREmitCTransforms
  InsertIncludes.cpp

  DEPENDS
  MLIREmitC
  MLIREmitCPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
