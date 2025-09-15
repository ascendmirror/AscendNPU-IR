set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(BISHENGIR_DIALECT_LIBS GLOBAL PROPERTY BISHENGIR_DIALECT_LIBS)
get_property(BISHENGIR_CONVERSION_LIBS GLOBAL PROPERTY BISHENGIR_CONVERSION_LIBS)

set(LIBS
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRMlirOptMain
  MLIRPass
  ${BISHENGIR_DIALECT_LIBS}
  ${BISHENGIR_CONVERSION_LIBS}
)

add_mlir_tool(bishengir-opt
  bishengir-opt.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(bishengir-opt PRIVATE ${LIBS})
if(NOT BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  target_link_libraries(bishengir-opt PRIVATE ${BISHENG_IR_LIBS})
endif()

llvm_update_compile_flags(bishengir-opt)
mlir_check_all_link_libraries(bishengir-opt)
