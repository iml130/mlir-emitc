add_mlir_library(MLIRTargetCpp
  Cpp/TranslateToCpp.cpp
  Cpp/TranslateToCppRegistration.cpp
  Cpp/TranslateToCRegistration.cpp
  Cpp/TranslationFlags.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/emitc/Target/Cpp

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIREmitC
  MLIRIR
  MLIRStandard
  )
