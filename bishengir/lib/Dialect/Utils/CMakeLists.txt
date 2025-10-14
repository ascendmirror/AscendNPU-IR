if(NOT BISHENGIR_BUILD_STANDALONE_IR_ONLY)
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
