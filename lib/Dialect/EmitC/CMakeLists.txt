add_subdirectory(Conversion)
add_subdirectory(IR)
add_subdirectory(Transforms)

add_mlir_library(MLIREmitCPipelines
  Pipelines.cpp

  DEPENDS
  MLIREmitCIncGen
  MLIREmitCPassIncGen
  MLIREmitCConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
