add_subdirectory(Annotation)

if(MLIR_ENABLE_EXECUTION_ENGINE)
  add_subdirectory(ExecutionEngine)
endif()

add_subdirectory(HACC)
add_subdirectory(HFusion)
add_subdirectory(HIVM)
add_subdirectory(MathExt)
add_subdirectory(MemRef)
add_subdirectory(MemRefExt)
add_subdirectory(SCF)
add_subdirectory(Symbol)
add_subdirectory(Tensor)

if(BISHENGIR_ENABLE_TORCH_CONVERSIONS)
  add_subdirectory(Torch)
endif()
