add_bishengir_dialect_library(BiShengIRHIVMPipelines
  ConvertToHIVMPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM

  LINK_LIBS PUBLIC
  BiShengIRHFusionToHIVM
  BiShengIRHIVMTransforms
  BiShengIRTensorToHIVM
  MLIRPass
)