if(${IREE_ENABLE_EMITC})
  # Dialects from iree::tools::init_mlir_passes_and_dialects
  set(dialect_libs
    MLIRAffineOps
    MLIRAffineTransforms
    MLIRGPU
    MLIRIR
    MLIRLLVMIR
    MLIRLinalgOps
    MLIRLinalgTransforms
    MLIRQuant
    MLIRSCF
    MLIRSCFTransforms
    MLIRSDBM
    MLIRSPIRV
    MLIRSPIRVTransforms
    MLIRStandardOps
    MLIRShape
    MLIRTransforms
    MLIRVector
    )
  # Passes from iree::tools::init_mlir_passes_and_dialects
  set(conversion_libs
    MLIRGPUToSPIRVTransforms
    MLIRLinalgToLLVM
    MLIRLinalgToSPIRVTransforms
    MLIRSCFToGPU
    MLIRStandardToSPIRVTransforms
    )
else()
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
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
  )

add_llvm_executable(emitc-opt
  emitc-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(emitc-opt PRIVATE ${LIBS})
llvm_update_compile_flags(emitc-opt)

mlir_check_all_link_libraries(emitc-opt)
