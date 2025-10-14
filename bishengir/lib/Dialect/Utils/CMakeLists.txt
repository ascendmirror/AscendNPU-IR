set(BS_FEAT_LINK_LIBS)
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
  list(APPEND BS_FEAT_LINK_LIBS BiShengIRLinalgDialectExt)
endif()

add_bishengir_library(BiShengIRDialectUtils
  OpInterfaceUtils.cpp
  ReachabilityAnalyzer.cpp
  UnionFind.cpp
  Util.cpp
  ValueDependencyAnalyzer.cpp

  DEPENDS
  BiShengIRHIVMEnumsIncGen

  LINK_LIBS PUBLIC
  BiShengIRAnnotationDialect
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  BiShengIRHFusionDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  BiShengIRMemRefDialect
  BiShengIRTensorDialect
  ${BS_FEAT_LINK_LIBS}
)
