if(${IREE_ENABLE_EMITC})
  # Dialects from iree::tools::init_mlir_passes_and_dialects
  set(dialect_libs
    MLIRAffine
    MLIRAffineTransforms
    MLIRGPU
    MLIRIR
    MLIRLLVMIR
    MLIRLinalg
    MLIRLinalgTransforms
    MLIRQuant
    MLIRSCF
    MLIRSCFTransforms
    MLIRSDBM
    MLIRSPIRV
    MLIRSPIRVTransforms
    MLIRShape
    MLIRStandard
    MLIRTosa
    MLIRTosaTransforms
    MLIRTransforms
    MLIRVector
    )
  # Passes from iree::tools::init_mlir_passes_and_dialects
  set(conversion_libs
    MLIRGPUToSPIRV
    MLIRLinalgToLLVM
    MLIRLinalgToSPIRV
    MLIRSCFToGPU
    MLIRStandardToSPIRV
    )
else()
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
endif()

if(${EMITC_ENABLE_HLO})
  set(HLO_LIBS
    MLIRMHLOToEmitC
    MLIRMHLORegionOpsToEmitC
    MLIREmitCConversion
    MhloRegisterDialects
    AllMhloPasses
    )
  set(HLO_LIBS_DEPS
    MLIREmitCMHLOPassIncGen
    MLIREmitCPassIncGen
    MLIRLmhloPassIncGen
    MLIRMhloPassIncGen
    )
else()
  unset(HLO_LIBS)
  unset(HLO_LIBS_DEPS)
endif()

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIREmitC
  MLIRSCFToEmitC
  MLIRStdToEmitC
  MLIRTensorToEmitC
  MLIRTosaToEmitC
  ${HLO_LIBS}
  )

add_llvm_executable(emitc-opt
  emitc-opt.cpp

  DEPENDS
  ${LIBS}
  ${HLO_LIBS_DEPS}
  )
target_link_libraries(emitc-opt PRIVATE ${LIBS})
llvm_update_compile_flags(emitc-opt)

#mlir_check_all_link_libraries(emitc-opt)
