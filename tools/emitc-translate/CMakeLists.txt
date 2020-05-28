set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_tool(emitc-translate
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
  MLIRTargetCppRegistration
  MLIRTargetCpp
  )

mlir_check_link_libraries(emitc-translate)
