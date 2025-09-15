add_bishengir_dialect_library(BiShengIRHIVMDialect
  HIVMDialect.cpp
  HIVMOps.cpp
  HIVMSynchronizationOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM

  DEPENDS
  BiShengIRHIVMDialectIncGen
  BiShengIRHIVMOpIncGen
  BiShengIRHIVMSynchronizationOpsIncGen
  BiShengIRHIVMEnumsIncGen
  BiShengIRHIVMAttributesIncGen
)