add_mlir_library(MLIRTargetCpp
  Cpp/TranslateToCpp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/Cpp
  )
target_link_libraries(MLIRTargetCpp
  PUBLIC
  MLIREmitC
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  )
add_mlir_library(MLIRTargetCppRegistration
  Cpp/TranslateToCppRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/Cpp
  )
target_link_libraries(MLIRTargetCppRegistration
  PUBLIC
  MLIRTargetCpp
  )