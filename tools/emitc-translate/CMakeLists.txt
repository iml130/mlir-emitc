set(LLVM_LINK_COMPONENTS
  Support
  )
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
else()
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
endif()

add_llvm_executable(emitc-translate
  emitc-translate.cpp
  )
llvm_update_compile_flags(emitc-translate)
target_link_libraries(emitc-translate
  PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  MLIREmitC
  MLIRTargetCpp
  )

mlir_check_link_libraries(emitc-translate)
