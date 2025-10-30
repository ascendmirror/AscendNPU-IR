add_bishengir_library(BiShengIRTransform
  CanonicalizeModule.cpp
  DeadFunctionElimination.cpp
  ExternalCanonicalizer.cpp
  LowerToCPUBackend.cpp
  Transforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Transforms

  DEPENDS
  BiShengIRTransformsPassIncGen

  LINK_LIBS PUBLIC
)