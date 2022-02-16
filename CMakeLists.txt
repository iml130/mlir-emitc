cmake_minimum_required(VERSION 3.13.4)
project(mlir-emitc LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

include(CMakeDependentOption)

#-------------------------------------------------------------------------------
# Options and definitions
#-------------------------------------------------------------------------------

option(EMITC_BUILD_EMBEDDED "Build EmitC as part of another project" OFF)
option(EMITC_ENABLE_HLO "Enables building MLIR-HLO." ON)
option(EMITC_TOSA_USE_EIGEN "Enables use of Eigen library for some TOSA Ops." OFF)
option(EMITC_INCLUDE_TESTS "Generate build targets for the MLIR EmitC unit tests." ON)
cmake_dependent_option(EMITC_TOSA_TEST_EIGEN "Enables testing of Eigen library for some TOSA Ops." ON "EMITC_INCLUDE_TESTS;EMITC_TOSA_USE_EIGEN" OFF)
# TODO: Set to MLIR or LLVM default
#       ${LLVM_INCLUDE_TESTS})

if(EMITC_ENABLE_HLO)
  message(STATUS "Building EmitC with MLIR-HLO conversions")
  add_definitions(-DEMITC_BUILD_HLO)
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM configuration
#-------------------------------------------------------------------------------

# Find MLIR if we are building standalone. If building as part of another
# project, let it handle the MLIR dependency. The dependent project might
# use a bundled version of MLIR instead of installing, for instance.
if(NOT EMITC_BUILD_EMBEDDED)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
endif()

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# TODO: With LLVM_ENABLE_WARNINGS set to ON in HandleLLVMOptions, the
# compilation of googletest fails. As a workaround, we remove the
# `-Wcovered-switch-default` flag.
string(REPLACE "-Wcovered-switch-default" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(EMITC_MAIN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include )
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# Dependent projects and dependencies
#-------------------------------------------------------------------------------

# Configure MHLO if we are building standalone. If building as part of
# another, let it handle the submodule and includes.
if(EMITC_ENABLE_HLO AND NOT EMITC_BUILD_EMBEDDED)
  add_subdirectory(third_party/mlir-hlo EXCLUDE_FROM_ALL)
  include_directories(${mlir-hlo_SOURCE_DIR}/include)
  include_directories(${mlir-hlo_BINARY_DIR}/include)
endif()

# Optional Eigen dependency for some TOSA Ops
if(EMITC_TOSA_USE_EIGEN)
  find_package(Eigen3 3.3.1 NO_MODULE)
  if(NOT TARGET Eigen3::Eigen)
    message(FATAL_ERROR "Should build with Eigen, but Eigen was not found.")
  endif()
endif()

# Dependency on GoogleTest. Used to unit test the reference implementation.
if(EMITC_INCLUDE_TESTS)
  include(third_party/cmake-scripts/code-coverage.cmake)

  # With `LLVM_INSTALL_UTILS` set to `ON`, the LLVM/MLIR build dir contains a
  # gtest, added in llvm/utils/unittest/CMakeLists.txt. However, LLVM ships
  # gooletest version 1.8, whereas we require a more recent version. Therefore,
  # we require to use an installed MLIR if unittests should be build.
  if(TARGET gtest)
    message(FATAL_ERROR
      "Unitests can only be build if MLIR is installed, but MLIR_DIR seems to "
      "point to an MLIR build directory.")
  endif()
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include/emitc)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(reference-implementation)
