set(LLVM_LINK_COMPONENTS
  Support
  )

set(LIBS
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRMlirOptMain
  MLIRPass
)

add_mlir_tool(bishengir-opt
  bishengir-opt.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(bishengir-opt PRIVATE ${LIBS})
target_link_libraries(bishengir-opt PRIVATE ${BISHENG_IR_LIBS})

llvm_update_compile_flags(bishengir-opt)
mlir_check_all_link_libraries(bishengir-opt)
