add_bishengir_dialect_library(BiShengIRHFusionDialect
  HFusionDialect.cpp
  HFusionImpl.cpp
  HFusionOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  DEPENDS
  BiShengIRSymbolIncGen
  BiShengIRHFusionAttributesIncGen
  BiShengIRHFusionEnumsIncGen
  BiShengIRHFusionOpsIncGen
  BiShengIRHFusionStructuredOpsIncGen

  LINK_LIBS PUBLIC
  BiShengIRLinalgDialectExt
  BiShengIRSymbolDialect
  BiShengIRHACCDialect
  BiShengIRHMAPDialect
  BiShengIRMathExtDialect
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRSideEffectInterfaces
  )
