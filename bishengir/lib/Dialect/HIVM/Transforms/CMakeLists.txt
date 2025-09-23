add_bishengir_dialect_library(BiShengIRHIVMTransforms
  ConvertToHIVMOp.cpp
  TritonGlobalKernelArgsToHIVMOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Transforms
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Utils
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Symbol

  DEPENDS
  BiShengIRHIVMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRArithDialect
  MLIRArithUtils
  BiShengIRAnnotationDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationTransformOps
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  BiShengIRHACCUtils
  BiShengIRHACCDialect
  BiShengIRHIVMDialect
  BiShengIRHIVMUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  BiShengIRMemRefExtDialect
  BiShengIRDialectUtils
  )
