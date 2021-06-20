add_subdirectory(Conversion)
add_subdirectory(Transforms)

add_mlir_library(MLIREmitCPipelines
  Pipelines.cpp

  DEPENDS
  MLIREmitC
  MLIREmitCPassIncGen
  MLIREmitCConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
