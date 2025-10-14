if(NOT BISHENGIR_BUILD_STANDALONE_IR_ONLY)
    add_subdirectory(Dialect)
    add_subdirectory(RegisterEverything)
endif()