set(LLVM_OPTIONAL_SOURCES
  MHLOToEmitC.cpp
  SCFToEmitC.cpp
  StdToEmitC.cpp
)

add_mlir_library(MLIRMHLOToEmitC
  MHLOToEmitC.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

add_mlir_library(MLIRSCFToEmitC
  SCFToEmitC.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

add_mlir_library(MLIRStdToEmitC
  StdToEmitC.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
