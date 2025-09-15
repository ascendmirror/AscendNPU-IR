// REQUIRES: bishengir_standalone_ir_build
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect -mlir-print-op-generic %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @pointer_cast
func.func @pointer_cast() {
  %c0_i64 = arith.constant 0 : i64
  %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16x16xf16>
  return
}