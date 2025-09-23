add_bishengir_dialect_library(BiShengIRArithShardingInterface
  MeshShardingInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRShardingInterface
)
