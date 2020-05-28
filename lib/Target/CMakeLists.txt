add_mlir_library(MLIRTargetCpp
  Cpp/TranslateToCpp.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/Target/Cpp

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIREmitC
  MLIRIR
  MLIRStandardOps
  )

add_mlir_library(MLIRTargetCppRegistration
  Cpp/TranslateToCppRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/Target/Cpp

  LINK_LIBS PUBLIC
  MLIRTargetCpp
  )
