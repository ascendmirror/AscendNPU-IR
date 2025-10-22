// RUN: bishengir-opt %s --split-input-file --cse --canonicalize --memref-forward-unit-memory --cse --canonicalize | FileCheck %s

// CHECK-LABEL: @reinterpret_cast_and_vcast(
// CHECK: %[[VAL_2:.*]] = memref.reinterpret_cast
// CHECK: hivm.hir.vcast ins(%[[VAL_2]] : memref<1xbf16, strided<[1], offset: ?>>)
// CHECK: return
module {
  func.func @reinterpret_cast_and_vcast(%arg0: memref<?xbf16>, %arg1: index) {
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
    %0 = memref.load %reinterpret_cast[%c0] : memref<1xbf16, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1xbf16>
    memref.store %0, %alloc[%c0] : memref<1xbf16>
    %alloc_0 = memref.alloc() : memref<1xf32>
    hivm.hir.vcast ins(%alloc : memref<1xbf16>) outs(%alloc_0 : memref<1xf32>)
    return
  }
}

// -----
module {
  func.func @scalar_sqrt_forward(%arg0: memref<?xf32>, %arg1: index) {
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
    %0 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1xf32>
    memref.store %0, %alloc[%c0] : memref<1xf32>
    %alloc_0 = memref.alloc() : memref<1xf32>
    %alloc_1 = memref.alloc() : memref<1xf32>
    hivm.hir.vln ins(%alloc : memref<1xf32>) outs(%alloc_0 : memref<1xf32>)
    hivm.hir.vexp ins(%alloc : memref<1xf32>) outs(%alloc_1 : memref<1xf32>)
    return
  }
}