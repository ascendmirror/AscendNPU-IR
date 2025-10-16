// RUN: bishengir-opt %s -hivm-aggregated-decompose-op="decompose-phase=after-hivm-align" -split-input-file -verify-diagnostics | FileCheck %s  --check-prefix=AFTERALIGN

// AFTERALIGN-LABEL: func @test_static_concat_last_dim
func.func @test_static_concat_last_dim() -> memref<136x4096xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<136x2048xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<136x2048xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<136x4096xf32>
  // CHECK:  %[[SUBVIEW:.*]] = memref.subview %alloc_1[0, 0] [136, 2048] [1, 1] : memref<136x4096xf32> to memref<136x2048xf32, strided<[4096, 1]>>
  // CHECK:  hivm.hir.copy ins(%[[alloc:.*]] : memref<136x2048xf32>) outs(%[[subview:.*]] : memref<136x2048xf32, strided<[4096, 1]>>)
  // CHECK:  %[[SUBVIEW_2:.*]] = memref.subview %alloc_1[0, 2048] [136, 2048] [1, 1] : memref<136x4096xf32> to memref<136x2048xf32, strided<[4096, 1], offset: 2048>>
  // CHECK:  hivm.hir.copy ins(%[[alloc_0:.*]] : memref<136x2048xf32>) outs(%[[subview_2:.*]] : memref<136x2048xf32, strided<[4096, 1], offset: 2048>>)
  hivm.hir.vconcat dim(1) ins(%alloc, %alloc_0 : memref<136x2048xf32>, memref<136x2048xf32>) outs(%alloc_1 : memref<136x4096xf32>)
  return %alloc_1 : memref<136x4096xf32>
}

// -----

// AFTERALIGN-LABEL: func @test_not_insert_slice_pattern_vconcat_0
// CHECK:  hivm.hir.copy
// CHECK:  hivm.hir.copy
// CHECK:  hivm.hir.copy
func.func @test_not_insert_slice_pattern_vconcat_0(%arg0: memref<?x1xf32>, %arg1: memref<32x1xf32>, %arg2: memref<?x1xf32>, %arg4: i32) -> memref<?x1xf32> {
  %0 = arith.index_cast %arg4 : i32 to index
  %alloc = memref.alloc(%0) : memref<?x1xf32>
  hivm.hir.vconcat dim(0) {hivm.insert_slice_source_index = 1 : i64} ins(%arg0, %arg1, %arg2 : memref<?x1xf32>, memref<32x1xf32>, memref<?x1xf32>) outs(%alloc : memref<?x1xf32>)
  return %alloc : memref<?x1xf32>
}

// -----

// AFTERALIGN-LABEL: func @test_insert_slice_pattern_vconcat_0(
// CHECK-SAME: %[[arg0:.*]]: memref<1024x8xf32>, %[[arg1:.*]]: memref<32x1xf32, strided<[8, 1]>>
// CHECK: %[[subview:.*]] = memref.subview %[[arg0]]
// CHECK: hivm.hir.copy ins(%[[arg1]]: memref<32x1xf32, strided<[8, 1]>> outs(%[[subview]]
// CHECK-NOT: hivm.hir.copy
func.func @test_insert_slice_pattern_vconcat_0(%arg0: memref<1024x8xf32>, %arg1: memref<32x1xf32, strided<[8, 1]>>, %arg2: i32, %arg3: i32, %arg4: i32) {
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg3 : i32 to index
  %3 = arith.index_cast %arg4 : i32 to index
  %subview0 = memref.subview %arg0[0, 0] [%1, 1] [1, 1] : memref<1024x8xf32> to memref<?x1xf32, strided<[8, 1]>>
  %subview1 = memref.subview %arg0[%2, 0] [%3, 1] [1, 1] : memref<1024x8xf32> to memref<?x1xf32, strided<[8, 1], offset: ?>>
  %alloc = memref.alloc() : memref<1024x8xf32>
  %subview2 = memref.subview %alloc[0, 0] [1024, 1] [1, 1] : memref<1024x8xf32> to memref<1024x1xf32, strided<[8, 1]>>
  hivm.hir.vconcat dim(0) {hivm.insert_slice_source_index = 1 : i64} ins(%subview0, %arg1, %subview1 : memref<?x1xf32, strided<[8, 1]>>, memref<32x1xf32, strided<[8, 1]>>, memref<?x1xf32, strided<[8, 1], offset: ?>>) 
                                                                     outs(%subview2 : memref<1024x1xf32, strided<[8, 1]>>)
  return
}

