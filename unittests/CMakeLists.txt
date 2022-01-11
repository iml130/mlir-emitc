add_executable(MLIREmitCTests "")
target_sources(MLIREmitCTests
  PRIVATE
    emitc_mhlo.cpp
    emitc_arith.cpp
    emitc_std.cpp
    emitc_tensor.cpp
    emitc_tosa_eigen.cpp
    emitc_tosa.cpp
    emitc_types.cpp
)

target_include_directories(MLIREmitCTests
  PRIVATE ${gtest_SOURCE_DIR}/include
  PRIVATE ${gmock_SOURCE_DIR}/include
)

target_link_libraries(MLIREmitCTests PRIVATE gtest_main gtest)

if(EMITC_TOSA_TEST_EIGEN)
  add_executable(MLIREmitCEigenTests "")
  target_sources(MLIREmitCEigenTests
    PRIVATE
      emitc_tosa_eigen.cpp
  )

  target_compile_definitions(MLIREmitCEigenTests PRIVATE EMITC_TOSA_USE_EIGEN)

  target_include_directories(MLIREmitCEigenTests
    PRIVATE ${gtest_SOURCE_DIR}/include
    PRIVATE ${gmock_SOURCE_DIR}/include
  )

  target_link_libraries(MLIREmitCEigenTests PRIVATE Eigen3::Eigen gtest_main gtest)
endif()
