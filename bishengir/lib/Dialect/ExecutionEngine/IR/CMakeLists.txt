add_bishengir_dialect_library(MLIRExecutionEngineDialect
  ExecutionEngineDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/ExecutionEngine

  DEPENDS
  MLIRExecutionEngineAttributesIncGen
)
