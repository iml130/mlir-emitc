add_mlir_translation_library(MLIRTargetCpp
  TranslateToCpp.cpp
  TranslateToCppRegistration.cpp
  TranslateToCRegistration.cpp
  TranslationFlags.cpp

  ADDITIONAL_HEADER_DIRS
  ${EMITC_MAIN_INCLUDE_DIR}/emitc/Target/Cpp

  LINK_LIBS PUBLIC
  MLIREmitC
  MLIRIR
  MLIRSCF
  MLIRStandard
  )
