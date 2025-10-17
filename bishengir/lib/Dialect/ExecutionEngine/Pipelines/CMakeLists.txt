get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_bishengir_dialect_library(MLIRExecutionEnginePipelines
  ExecutionEnginePipelines.cpp

  LINK_LIBS PUBLIC
  ${conversion_libs}
  ${dialect_libs}
)
