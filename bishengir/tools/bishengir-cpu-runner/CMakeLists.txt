if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
)

get_property(PUBLIC_BISHENGIR_DIALECT_LIBS GLOBAL PROPERTY BISHENGIR_DIALECT_OPEN_SOURCE_LIBS)
get_property(PUBLIC_BISHENGIR_LIBS GLOBAL PROPERTY BISHENGIR_OPEN_SOURCE_LIBS)
get_property(PRIVATE_BISHENGIR_DIALECT_LIBS GLOBAL PROPERTY BISHENGIR_DIALECT_LIBS)
get_property(PRIVATE_BISHENGIR_LIBS GLOBAL PROPERTY BISHENGIR_LIBS)

add_mlir_tool(bishengir-cpu-runner
  bishengir-cpu-runner.cpp
)
llvm_update_compile_flags(bishengir-cpu-runner)
target_link_libraries(bishengir-cpu-runner PRIVATE
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRIR
  MLIRJitRunner
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRParser
  MLIRTargetLLVMIRExport
  MLIRSupport
  ${DIALECT_LIBS}
  ${PUBLIC_BISHENGIR_DIALECT_LIBS}
  ${PUBLIC_BISHENGIR_LIBS}
  ${PRIVATE_BISHENGIR_DIALECT_LIBS}
  ${PRIVATE_BISHENGIR_LIBS}
)

export_executable_symbols(bishengir-cpu-runner)
