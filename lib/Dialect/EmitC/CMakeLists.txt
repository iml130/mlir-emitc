add_subdirectory(Transforms)

add_mlir_library(MLIREmitCPipelines
  Pipelines.cpp

  DEPENDS
  MLIREmitCDialect
  MLIREmitCPassIncGen
  MLIREmitCConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
