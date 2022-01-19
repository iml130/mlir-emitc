get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(${EMITC_ENABLE_HLO})
  set(HLO_LIBS
    MLIRMHLOToEmitC
    MLIRMHLORegionOpsToEmitC
    MhloRegisterDialects
    )
  set(HLO_LIBS_DEPS
    MLIREmitCConversionPassIncGen
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
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIREmitC
  MLIRArithToEmitC
  MLIRStdToEmitC
  MLIRTensorToEmitC
  MLIRTosaToEmitC
  MLIREmitCTransforms
  MLIREmitCPipelines
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
