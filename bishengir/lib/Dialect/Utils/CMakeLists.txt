add_bishengir_library(BiShengIRDialectUtils
  OpInterfaceUtils.cpp
  ReachabilityAnalyzer.cpp
  UnionFind.cpp
  Util.cpp
  ValueDependencyAnalyzer.cpp

  DEPENDS
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen
  BiShengIRHIVMEnumsIncGen
  BiShengIRHIVMAttributesIncGen

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
  BiShengIRLinalgDialectExt
)
