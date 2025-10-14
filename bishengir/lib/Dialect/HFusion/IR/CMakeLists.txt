set(BS_LINALG_EXT_LINK_LIBS)
set(BS_HACC_LINK_LIBS)
if(NOT BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  add_mlir_dialect_library(BiShengIRLinalgDialectExt
    LinalgExtensions.cpp
 
    PARTIAL_SOURCES_INTENDED
 
    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
 
    DEPENDS
    MLIRLinalgInterfacesIncGen
    MLIRLinalgOpsAttributesIncGen
    MLIRLinalgOpsEnumsIncGen
    MLIRLinalgOpsIncGen
    MLIRLinalgStructuredOpsIncGen
 
    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
  )
  list(APPEND BS_LINALG_EXT_LINK_LIBS BiShengIRLinalgDialectExt)
  add_bishengir_dialect_library(BiShengIRHACCDialect
    HACCDialect.cpp
    HACCInterfaces.cpp

    ADDITIONAL_HEADER_DIRS
    ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HACC

    DEPENDS
    BiShengIRHACCEnumsIncGen
    BiShengIRHACCAttributesIncGen
    BiShengIRHACCInterfacesIncGen
    BiShengIRNPUTargetSpecIncGen

    LINK_LIBS PUBLIC
    MLIRLLVMDialect
    MLIRFunctionInterfaces
    MLIRSideEffectInterfaces
    MLIRDestinationStyleOpInterface
  )
  list(APPEND BS_HACC_LINK_LIBS BiShengIRHACCDialect)
endif()

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
  ${BS_LINALG_EXT_LINK_LIBS}
  BiShengIRSymbolDialect
  ${BS_HACC_LINK_LIBS}
  BiShengIRMathExtDialect
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRSideEffectInterfaces
  )
