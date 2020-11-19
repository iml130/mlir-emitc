add_custom_target(MLIREmitCUnitTests)
set_target_properties(MLIREmitCUnitTests PROPERTIES FOLDER "MLIR EmitC Tests")

function(add_mlir_unittest test_dirname)
  add_unittest(MLIREmitCUnitTests ${test_dirname} ${ARGN})
endfunction()

set(LLVM_LINK_COMPONENTS
  Support
  )

add_mlir_unittest(MLIREmitCAllTests emitc_mhlo.cpp emitc_std.cpp emitc_tensor.cpp)

target_include_directories(MLIREmitCAllTests
  PRIVATE ${gtest_SOURCE_DIR}/include
  PRIVATE ${gmock_SOURCE_DIR}/include
)
