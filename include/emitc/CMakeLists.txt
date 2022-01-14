add_subdirectory(Dialect)

#-------------------------------------------------------------------------------
# EmitC Reference Implementation
#-------------------------------------------------------------------------------

set(EMITC_REF_SRCS
  emitc_arith.h
  emitc_core_ops.h
  emitc_mhlo.h
  emitc_std.h
  emitc_tensor.h
  emitc_tosa.h
  emitc_types.h
  emitc_utility.h
)

add_library(EmitCRefImpl INTERFACE)
target_sources(EmitCRefImpl
  INTERFACE
    ${EMITC_REF_SRCS}
)
set_target_properties(EmitCRefImpl PROPERTIES LINKER_LANGUAGE CXX)

if(EMITC_TOSA_USE_EIGEN)
    add_library(EmitCRefImpl_Eigen INTERFACE)
    target_sources(EmitCRefImpl_Eigen
    INTERFACE
      ${EMITC_REF_SRCS}
      emitc_tosa_eigen.h
    )
    target_compile_definitions(EmitCRefImpl_Eigen INTERFACE EMITC_TOSA_USE_EIGEN)
    set_target_properties(EmitCRefImpl_Eigen PROPERTIES LINKER_LANGUAGE CXX)
endif()
