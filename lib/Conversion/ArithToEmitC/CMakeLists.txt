add_mlir_library(MLIRArithToEmitCExt
  ArithToEmitC.cpp

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
