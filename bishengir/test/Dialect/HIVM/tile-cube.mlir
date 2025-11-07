// RUN: bishengir-opt %s -tile-cube-vector-loop="tile-mix-cube-loop=2" -split-input-file | FileCheck %s --check-prefix=CHECK-CUBE

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_cube(%arg0: memref<256x192xbf16>, %arg1: memref<256x192xbf16>, %arg2: memref<2x256x256xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) {
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<256x192xbf16>
    scf.for %arg7 = %c0 to %arg6 step %c1 {
      %alloc = memref.alloc() : memref<256x192xbf16>
      %subview = memref.subview %arg0[0, 0] [%arg5, 192] [1, 1] : memref<256x192xbf16> to memref<?x192xbf16, strided<[192, 1]>>
      %subview_0 = memref.subview %alloc[0, 0] [%arg5, 192] [1, 1] : memref<256x192xbf16> to memref<?x192xbf16, strided<[192, 1]>>
      hivm.hir.load ins(%subview : memref<?x192xbf16, strided<[192, 1]>>) outs(%subview_0 : memref<?x192xbf16, strided<[192, 1]>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %true : i1 may_implicit_transpose_with_last_axis = false
      %1 = bufferization.to_tensor %alloc restrict writable : memref<256x192xbf16>
      %2 = tensor.empty() : tensor<256x256xf32>
      %3 = hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%0, %1, %true, %arg3, %arg4, %arg5 : tensor<256x192xbf16>, tensor<256x192xbf16>, i1, index, index, index) outs(%2 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %subview_1 = memref.subview %arg2[%arg7, 0, 0] [1, 256, 256] [1, 1, 1] : memref<2x256x256xf32> to memref<1x256x256xf32, strided<[65536, 256, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview_1 [[0, 1], [2]] : memref<1x256x256xf32, strided<[65536, 256, 1], offset: ?>> into memref<256x256xf32, strided<[256, 1], offset: ?>>
      hivm.hir.fixpipe {enable_nz2nd} ins(%3 : tensor<256x256xf32>) outs(%collapse_shape : memref<256x256xf32, strided<[256, 1], offset: ?>>)
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
    return
  }
}

