add_bishengir_library(BiShengIRMemRefTransforms
  DeadStoreElimination.cpp
  FoldAllocReshape.cpp
  ForwardUnitMemory.cpp
  RemoveRedundantCopy.cpp
  PropagateReshape/PropagateMemrefCollapseDown.cpp
  PropagateReshape/PropagateMemrefExpandUp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/MemRef/Transforms

  DEPENDS
  MLIRMemRefPassIncGen

  LINK_LIBS PUBLIC
  MLIRMemRefDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)