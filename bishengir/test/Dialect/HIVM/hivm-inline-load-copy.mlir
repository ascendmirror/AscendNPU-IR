// RUN: bishengir-opt --hivm-inline-load-copy -split-input-file %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_load_copy
func.func @test_load_copy(%arg0: memref<5x128xf32>, %arg1: memref<5x128xf32>) {
  // CHECK-NOT: hivm.hir.copy
  // CHECK: hivm.hir.load ins(%arg0 : memref<5x128xf32>) outs(%arg1 : memref<5x128xf32>)
  %empty0 = memref.alloc() : memref<5x128xf32>
  hivm.hir.load ins(%arg0 : memref<5x128xf32>) outs(%empty0 : memref<5x128xf32>)
  hivm.hir.copy ins(%empty0 : memref<5x128xf32>) outs(%arg1 : memref<5x128xf32>)
  return
}