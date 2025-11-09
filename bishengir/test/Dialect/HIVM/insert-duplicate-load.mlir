// RUN: bishengir-opt -insert-duplicate-load %s -split-input-file -verify-diagnostics --canonicalize | FileCheck %s

// -----
// CHECK-LABEL: @insert_duplicate_load
func.func @insert_duplicate_load(%arg0 : memref<?xf16>, %arg1 : memref<16x16xf16>, %arg2 : tensor<16x16xf32>, %arg3 : index, %arg4 : index) -> tensor<16x16xf32> {
   %c0 = arith.constant 0 : i1
   %cst_0 = arith.constant 0 : index
   %cst_100 = arith.constant 100 : index
   %cst_200 = arith.constant 200 : index
   %alloc = memref.alloc() : memref<16x16xf16>
   %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%cst_0], sizes: [16, 16], strides: [512, 1] : memref<?xf16> to memref<16x16xf16, strided<[512, 1], offset: ?>>
   %subview_1 = memref.subview %reinterpret_cast[0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
   %subview_2 = memref.subview %alloc[0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16> to memref<?x?xf16, strided<[16, 1]>>
   hivm.hir.load ins(%subview_1 : memref<?x?xf16, strided<[512, 1], offset: ?>>) outs(%subview_2 : memref<?x?xf16, strided<[16, 1]>>) left_padding_num = %cst_0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
   %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
   // CHECK: %[[VAL1:.*]] = memref.alloc() : memref<16x16xf16>
   // CHECK-NEXT: %[[VAL2:.*]] = memref.reinterpret_cast %arg0 to offset: [%{{.*}}], sizes: [16, 16], strides: [512, 1] : memref<?xf16> to memref<16x16xf16, strided<[512, 1], offset: ?>>
   // CHECK-NEXT: %[[VAL3:.*]] = memref.subview %[[VAL2]][0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
   // CHECK-NEXT: %[[VAL4:.*]] = memref.subview %[[VAL1]][0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16> to memref<?x?xf16, strided<[16, 1]>>
   // CHECK-NEXT: hivm.hir.load ins(%[[VAL3]] : memref<?x?xf16, strided<[512, 1], offset: ?>>) outs(%[[VAL4]] : memref<?x?xf16, strided<[16, 1]>>) left_padding_num = %{{.*}} : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
   // CHECK-NEXT: %[[VAL5:.*]] = bufferization.to_tensor %[[VAL1]] restrict writable : memref<16x16xf16>

   // CHECK-NEXT: %[[VAL6:.*]] = memref.alloc() : memref<16x16xf16>
   // CHECK-NEXT: %[[VAL7:.*]] = memref.subview %[[VAL6]][0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16> to memref<?x?xf16, strided<[16, 1]>>
   // CHECK-NEXT: hivm.hir.load ins(%[[VAL3]] : memref<?x?xf16, strided<[512, 1], offset: ?>>) outs(%[[VAL7]] : memref<?x?xf16, strided<[16, 1]>>) left_padding_num = %{{.*}} : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
   // CHECK-NEXT: %[[VAL8:.*]] = bufferization.to_tensor %[[VAL6]] restrict writable : memref<16x16xf16>
   %1 = tensor.empty() : tensor<16x16xf16>
   %t = tensor.empty() : tensor<16x16xf16>
   // CHECK: %{{.*}} = hivm.hir.vadd ins(%[[VAL5]]
   %2 = hivm.hir.vadd ins(%0, %1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%t : tensor<16x16xf16>) -> tensor<16x16xf16>
   hivm.hir.store ins(%2 : tensor<16x16xf16>) outs(%arg1 : memref<16x16xf16>)
   %3 = tensor.empty() : tensor<16x16xf16>
   // CHECK: %{{.*}} = hivm.hir.mmadL1 ins(%[[VAL8]]
   %4 = hivm.hir.mmadL1 ins(%0, %3, %c0, %cst_0, %cst_0, %cst_0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%arg2 : tensor<16x16xf32>) -> tensor<16x16xf32>
   return %4 : tensor<16x16xf32>
}

// -----
// CHECK-LABEL: @insert_duplicate_load_SCF_FOR
func.func @insert_duplicate_load_SCF_FOR(%arg0 : memref<?xf16>, %arg1 : memref<16x16xf16>, %arg2 : tensor<16x16xf16>, %arg3 : index, %arg4 : index) -> tensor<16x16xf16> {
   %c0 = arith.constant 0 : i1
   %cst_0 = arith.constant 0 : index
   %cst_1 = arith.constant 1 : index
   %cst_100 = arith.constant 100 : index
   %cst_200 = arith.constant 200 : index
   %alloc = memref.alloc() : memref<16x16xf16>
   %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%cst_0], sizes: [16, 16], strides: [512, 1] : memref<?xf16> to memref<16x16xf16, strided<[512, 1], offset: ?>>
   %subview_1 = memref.subview %reinterpret_cast[0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
   %subview_2 = memref.subview %alloc[0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16> to memref<?x?xf16, strided<[16, 1]>>
   hivm.hir.load ins(%subview_1 : memref<?x?xf16, strided<[512, 1], offset: ?>>) outs(%subview_2 : memref<?x?xf16, strided<[16, 1]>>) left_padding_num = %cst_0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
   %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
   // CHECK: %[[VAL1:.*]] = memref.alloc() : memref<16x16xf16>
   // CHECK-NEXT: %[[VAL2:.*]] = memref.reinterpret_cast %arg0 to offset: [%{{.*}}], sizes: [16, 16], strides: [512, 1] : memref<?xf16> to memref<16x16xf16, strided<[512, 1], offset: ?>>
   // CHECK-NEXT: %[[VAL3:.*]] = memref.subview %[[VAL2]][0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
   // CHECK-NEXT: %[[VAL4:.*]] = memref.subview %[[VAL1]][0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16> to memref<?x?xf16, strided<[16, 1]>>
   // CHECK-NEXT: hivm.hir.load ins(%[[VAL3]] : memref<?x?xf16, strided<[512, 1], offset: ?>>) outs(%[[VAL4]] : memref<?x?xf16, strided<[16, 1]>>) left_padding_num = %{{.*}} : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
   // CHECK-NEXT: %[[VAL5:.*]] = bufferization.to_tensor %[[VAL1]] restrict writable : memref<16x16xf16>
   // CHECK-NEXT: %[[VAL6:.*]] = memref.alloc() : memref<16x16xf16>
   // CHECK-NEXT: %[[VAL7:.*]] = memref.subview %[[VAL6]][0, 0] [%arg3, %arg4] [1, 1] : memref<16x16xf16> to memref<?x?xf16, strided<[16, 1]>>
   // CHECK-NEXT: hivm.hir.load ins(%[[VAL3]] : memref<?x?xf16, strided<[512, 1], offset: ?>>) outs(%[[VAL7]] : memref<?x?xf16, strided<[16, 1]>>) left_padding_num = %{{.*}} : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
   // CHECK-NEXT: %[[VAL8:.*]] = bufferization.to_tensor %[[VAL6]] restrict writable : memref<16x16xf16>
   // CHECK-NEXT: %2:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[VAL9:.*]] = %[[VAL5]], %[[VAL10:.*]] = %[[VAL8]]) -> (tensor<16x16xf16>, tensor<16x16xf16>) {
   %result = scf.for %i = %cst_0 to %cst_100 step %cst_1 iter_args(%arg5 = %0) -> tensor<16x16xf16> {
       %1 = tensor.empty() : tensor<16x16xf16>
       %t = tensor.empty() : tensor<16x16xf16>
       // CHECK: %{{.*}} = hivm.hir.vadd ins(%[[VAL9]]
       %2 = hivm.hir.vadd ins(%arg5, %1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%t : tensor<16x16xf16>) -> tensor<16x16xf16>
       hivm.hir.store ins(%2 : tensor<16x16xf16>) outs(%arg1 : memref<16x16xf16>)
       %3 = tensor.empty() : tensor<16x16xf16>
       // CHECK: %{{.*}} = hivm.hir.mmadL1 ins(%[[VAL10]]
       %4 = hivm.hir.mmadL1 ins(%arg5, %3, %c0, %cst_0, %cst_0, %cst_0 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%arg2 : tensor<16x16xf16>) -> tensor<16x16xf16>
       // CHECK: scf.yield %{{.*}}, %{{.*}} : tensor<16x16xf16>, tensor<16x16xf16>
       scf.yield %4 : tensor<16x16xf16>
   }
   return %result : tensor<16x16xf16>
}