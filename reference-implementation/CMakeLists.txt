#-------------------------------------------------------------------------------
# EmitC Reference Implementation
#-------------------------------------------------------------------------------

set(EMITC_REF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(EMITC_REF_SRCS
  ${EMITC_REF_INCLUDE_DIR}/emitc/arith.h
  ${EMITC_REF_INCLUDE_DIR}/emitc/core_ops.h
  ${EMITC_REF_INCLUDE_DIR}/emitc/stablehlo.h
  ${EMITC_REF_INCLUDE_DIR}/emitc/tensor.h
  ${EMITC_REF_INCLUDE_DIR}/emitc/tosa.h
  ${EMITC_REF_INCLUDE_DIR}/emitc/types.h
  ${EMITC_REF_INCLUDE_DIR}/emitc/utility.h
)

add_library(EmitCRefImpl INTERFACE)
target_sources(EmitCRefImpl
  INTERFACE
    ${EMITC_REF_SRCS}
)
target_include_directories(EmitCRefImpl INTERFACE ${EMITC_REF_INCLUDE_DIR})

if(EMITC_TOSA_USE_EIGEN)
    add_library(EmitCRefImpl_Eigen INTERFACE)
    target_sources(EmitCRefImpl_Eigen
    INTERFACE
      ${EMITC_REF_SRCS}
      ${EMITC_REF_INCLUDE_DIR}/emitc/tosa_eigen.h
    )
    target_include_directories(EmitCRefImpl_Eigen INTERFACE ${EMITC_REF_INCLUDE_DIR})
    target_compile_definitions(EmitCRefImpl_Eigen INTERFACE EMITC_TOSA_USE_EIGEN)
endif()


#-------------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------------

if(EMITC_INCLUDE_TESTS)
  add_subdirectory(unittests)
endif()
