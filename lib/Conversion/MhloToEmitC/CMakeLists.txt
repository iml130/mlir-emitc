set(LLVM_OPTIONAL_SOURCES
  MHLOToEmitC.cpp
  MHLORegionOpsToEmitC.cpp
)

if(EMITC_ENABLE_HLO)
  add_mlir_library(MLIRMHLOToEmitC
    MHLOToEmitC.cpp

    DEPENDS
    MLIREmitCDialect
    MLIREmitCConversionPassIncGen
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
    MLIREmitCDialect
    MLIREmitCConversionPassIncGen
    MLIRhlo_opsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransformUtils
  )
endif()
