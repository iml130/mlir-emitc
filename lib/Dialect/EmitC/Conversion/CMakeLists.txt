set(LLVM_OPTIONAL_SOURCES
  MHLOToEmitC.cpp
  SCFToEmitC.cpp
  StdToEmitC.cpp
)

if(EMITC_ENABLE_HLO)
  add_mlir_library(MLIRMHLOToEmitC
    MHLOToEmitC.cpp

    DEPENDS
    MLIREmitCIncGen
    MLIREmitCPassIncGen
    MLIRhlo_opsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransformUtils
  )
endif()

add_mlir_library(MLIRSCFToEmitC
  SCFToEmitC.cpp

  DEPENDS
  MLIREmitCIncGen
  MLIREmitCPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

add_mlir_library(MLIRStdToEmitC
  StdToEmitC.cpp

  DEPENDS
  MLIREmitCIncGen
  MLIREmitCPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
