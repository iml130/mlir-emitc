set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    MLIRAffineTransformsTestPasses
    MLIRSPIRVTestPasses
    MLIRTestDialect
    MLIRTestIR
    MLIRTestPass
    MLIRTestTransforms
    )
endif()

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${test_libs}
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
  )

add_llvm_tool(emitc-opt
  emitc-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(emitc-opt PRIVATE ${LIBS})
llvm_update_compile_flags(emitc-opt)

mlir_check_all_link_libraries(emitc-opt)
