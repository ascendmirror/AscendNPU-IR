
#map = affine_map<()[s0] -> (256, s0)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
#map2 = affine_map<()[s0] -> (192, s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_cube(%arg0: memref<256x192xbf16>, %arg1: memref<256x192xbf16>, %arg2: memref<2x256x256xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) {
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<256x192xbf16>
    scf.for %arg7 = %c0 to %arg6 step %c1 {
      %alloc = memref.alloc() : memref<256x192xbf16>
      %1 = bufferization.to_tensor %alloc restrict writable : memref<256x192xbf16>
      %2 = tensor.empty() : tensor<256x256xf32>
      %subview = memref.subview %arg2[%arg7, 0, 0] [1, 256, 256] [1, 1, 1] : memref<2x256x256xf32> to memref<1x256x256xf32, strided<[65536, 256, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x256x256xf32, strided<[65536, 256, 1], offset: ?>> into memref<256x256xf32, strided<[256, 1], offset: ?>>
      scf.for %arg8 = %c0 to %c256 step %c128 {
        %subview_0 = memref.subview %arg0[0, 0] [%arg5, 192] [1, 1] : memref<256x192xbf16> to memref<?x192xbf16, strided<[192, 1]>>
        %subview_1 = memref.subview %alloc[0, 0] [%arg5, 192] [1, 1] : memref<256x192xbf16> to memref<?x192xbf16, strided<[192, 1]>>
        hivm.hir.load ins(%subview_0 : memref<?x192xbf16, strided<[192, 1]>>) outs(%subview_1 : memref<?x192xbf16, strided<[192, 1]>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %true : i1 may_implicit_transpose_with_last_axis = false
        %extracted_slice = tensor.extract_slice %1[%arg8, 0] [128, 192] [1, 1] : tensor<256x192xbf16> to tensor<128x192xbf16>
        %extracted_slice_2 = tensor.extract_slice %2[0, %arg8] [256, 128] [1, 1] : tensor<256x256xf32> to tensor<256x128xf32>
        %3 = affine.min #map()[%arg3]
        %4 = affine.min #map1(%arg8)[%arg5]
        %5 = affine.min #map2()[%arg4]
        %6 = hivm.hir.mmadL1 {b_transpose, cube_producer_to_fuse_0, fixpipe_already_inserted = true} ins(%0, %extracted_slice, %true, %3, %5, %4 : tensor<256x192xbf16>, tensor<128x192xbf16>, i1, index, index, index) outs(%extracted_slice_2 : tensor<256x128xf32>) -> tensor<256x128xf32>
        %subview_3 = memref.subview %collapse_shape[0, %arg8] [256, 128] [1, 1] : memref<256x256xf32, strided<[256, 1], offset: ?>> to memref<256x128xf32, strided<[256, 1], offset: ?>>
        hivm.hir.fixpipe {enable_nz2nd, op_to_tile_0_0} ins(%6 : tensor<256x128xf32>) outs(%subview_3 : memref<256x128xf32, strided<[256, 1], offset: ?>>)
      }
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
    return
  }
}

