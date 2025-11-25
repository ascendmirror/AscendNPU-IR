add_bishengir_dialect_library(BiShengIRMemRefCanonicalization
  Canonicalization.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/MemRef

  LINK_LIBS PUBLIC
  MLIRMemRefDialect
  MLIRIR
  MLIRDialect
)

