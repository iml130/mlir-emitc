set(LLVM_OPTIONAL_SOURCES
  emitc_mhlo.cpp
  emitc_arith.cpp
  emitc_std.cpp
  emitc_tensor.cpp
  emitc_tosa_eigen.cpp
  emitc_tosa.cpp
  emitc_types.cpp
)

add_custom_target(MLIREmitCUnitTests)
set_target_properties(MLIREmitCUnitTests PROPERTIES FOLDER "MLIR EmitC Tests")

function(add_mlir_unittest test_dirname)
  add_unittest(MLIREmitCUnitTests ${test_dirname} ${ARGN})
endfunction()

set(LLVM_LINK_COMPONENTS
  Support
  )

add_mlir_unittest(MLIREmitCAllTests emitc_mhlo.cpp emitc_arith.cpp emitc_std.cpp emitc_tensor.cpp emitc_tosa.cpp emitc_tosa_eigen.cpp emitc_types.cpp)

target_include_directories(MLIREmitCAllTests
  PRIVATE ${gtest_SOURCE_DIR}/include
  PRIVATE ${gmock_SOURCE_DIR}/include
)

if(EMITC_TOSA_TEST_EIGEN)
  add_mlir_unittest(MLIREmitCEigenTests emitc_tosa_eigen.cpp)

  target_compile_definitions(MLIREmitCEigenTests PRIVATE EMITC_TOSA_USE_EIGEN)

  target_include_directories(MLIREmitCEigenTests
    PRIVATE ${gtest_SOURCE_DIR}/include
    PRIVATE ${gmock_SOURCE_DIR}/include
    PRIVATE ${EIGEN3_INCLUDE_DIR}
  )
endif()
