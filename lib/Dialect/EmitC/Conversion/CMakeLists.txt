set(LLVM_OPTIONAL_SOURCES
  MHLOToEmitC.cpp
  MHLORegionOpsToEmitC.cpp
  StdToEmitC.cpp
  TensorToEmitC.cpp
  TosaToEmitC.cpp
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

  add_mlir_library(MLIRMHLORegionOpsToEmitC
    MHLORegionOpsToEmitC.cpp

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

add_mlir_library(MLIRTensorToEmitC
  TensorToEmitC.cpp

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

add_mlir_library(MLIRTosaToEmitC
  TosaToEmitC.cpp

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
