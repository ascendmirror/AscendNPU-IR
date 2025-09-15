# BiShengIR Project.

option(BISHENGIR_BUILD_EXAMPLES
  "Build the BiShengIR example programs. If OFF, just generate build targets." OFF)

set(BISHENG_IR_INSTALL_PATH "" CACHE STRING
      "install path for pre-built required for bishengir")

option(BISHENGIR_BUILD_STANDALONE_IR_ONLY
  "Experimental feature. If set, only build the IR in a standalone manner." OFF)

if(BISHENG_IR_INSTALL_PATH STREQUAL "")
  if(NOT BISHENGIR_BUILD_STANDALONE_IR_ONLY)
    message(FATAL_ERROR "Must provide the path to pre-built.")
  endif()
endif()

set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_MAIN_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
set(MLIR_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)

set(BISHENGIR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bishengir)
set(BISHENGIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(BISHENGIR_MAIN_INCLUDE_DIR ${BISHENGIR_SRC_DIR}/include)
set(BISHENGIR_INCLUDE_DIR ${BISHENGIR_BINARY_DIR}/bishengir/include)
set(BISHENGIR_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(BISHENG_IR_LIBS ${BISHENG_IR_INSTALL_PATH}/lib/libBiShengIR.so)
set(BISHENG_IR_YAML_GEN_EXE ${BISHENG_IR_INSTALL_PATH}/bin/bishengir-yaml-gen)

# Make sure that our source directory is on the current cmake module path so
# that we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0
  "${BISHENGIR_SRC_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)
include(AddBiShengIR)

configure_file(
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Config/bishengir-config.h.cmake
  ${BISHENGIR_INCLUDE_DIR}/bishengir/Config/bishengir-config.h)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_MAIN_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${BISHENGIR_MAIN_INCLUDE_DIR})
include_directories(${BISHENGIR_INCLUDE_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")

add_subdirectory(bishengir/include/bishengir)
add_subdirectory(bishengir/lib)
add_subdirectory(bishengir/tools)

if(BISHENGIR_BUILD_EXAMPLES)
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH}
        CACHE STRING "Ascend CANN install path")
  if(${ASCEND_HOME_PATH} STREQUAL "")
    message(FATAL_ERROR "Ascend CANN package is required for building example!")
  endif()
  add_subdirectory(examples)
endif()

if(MLIR_INCLUDE_TESTS)
  add_definitions(-DMLIR_INCLUDE_TESTS)
  add_subdirectory(bishengir/test)
endif()
