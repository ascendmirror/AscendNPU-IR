add_bishengir_dialect_library(BiShengIRAnnotationDialect
  AnnotationDialect.cpp
  AnnotationOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Annotation

  DEPENDS
  MLIRAnnotationOpsIncGen
)