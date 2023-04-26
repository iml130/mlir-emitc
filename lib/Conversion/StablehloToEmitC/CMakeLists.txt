if(EMITC_ENABLE_HLO)
  add_mlir_library(MLIRStablehloToEmitC
    StablehloToEmitC.cpp

    DEPENDS
    MLIREmitCDialect
    MLIREmitCConversionPassIncGen
    StablehloOpsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransformUtils
  )
endif()
