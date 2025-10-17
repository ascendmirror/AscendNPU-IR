get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_bishengir_dialect_library(MLIRExecutionEngineTransforms
  ConvertHIVMToUpstream.cpp
  CreateHostMain.cpp

  DEPENDS
  MLIRExecutionEngineTransformsIncGen

  LINK_LIBS PUBLIC
  ${dialect_libs}
  MLIRHACCUtils
  MLIRPass
)
