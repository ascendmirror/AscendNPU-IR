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
    %extracted_slice = tensor.extract_slice %1[0, 0] [%arg5, 192] [1, 1] : tensor<256x192xbf16> to tensor<?x192xbf16>
    %2 = bufferization.to_tensor %arg0 restrict writable : memref<256x192xbf16>
    %extracted_slice_0 = tensor.extract_slice %2[0, 0] [%arg5, 192] [1, 1] : tensor<256x192xbf16> to tensor<?x192xbf16>
    %3 = tensor.empty() : tensor<256x256xf32>
    %subview = memref.subview %arg2[%arg7, 0, 0] [1, 256, 256] [1, 1, 1] : memref<2x256x256xf32> to memref<1x256x256xf32, strided<[65536, 256, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x256x256xf32, strided<[65536, 256, 1], offset: ?>> into memref<256x256xf32, strided<[256, 1], offset: ?>>
    scf.for %arg8 = %c0 to %c256 step %c128 {
      %4 = affine.max affine_map<(d0) -> (0, d0)>(%arg8)
      %5 = affine.min affine_map<(d0)[s0] -> (s0, d0)>(%4)[%arg5]
      %6 = affine.min affine_map<(d0)[s0] -> (s0, d0 + 128)>(%arg8)[%arg5]
      %7 = affine.max affine_map<(d0, d1) -> (0, d0 - d1)>(%6, %5)
      %extracted_slice_1 = tensor.extract_slice %extracted_slice_0[0, 0] [%arg5, 192] [1, 1] : tensor<?x192xbf16> to tensor<?x192xbf16>
      %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, 0] [%arg5, 192] [1, 1] : tensor<?x192xbf16> to tensor<?x192xbf16>
      %8 = hivm.hir.load ins(%extracted_slice_1 : tensor<?x192xbf16>) outs(%extracted_slice_2 : tensor<?x192xbf16>) {cube_producer_to_fuse_0, lifted_load} pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %true : i1 may_implicit_transpose_with_last_axis = false -> tensor<?x192xbf16>
      %extracted_slice_3 = tensor.extract_slice %1[%arg8, 0] [128, 192] [1, 1] : tensor<256x192xbf16> to tensor<128x192xbf16>
      %inserted_slice = tensor.insert_slice %8 into %extracted_slice_3[%arg8, 0] [%7, 192] [1, 1] : tensor<?x192xbf16> into tensor<128x192xbf16>
      %extracted_slice_4 = tensor.extract_slice %3[0, %arg8] [256, 128] [1, 1] : tensor<256x256xf32> to tensor<256x128xf32>
      %9 = affine.min affine_map<()[s0] -> (256, s0)>()[%arg3]
      %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 128)>(%arg8)[%arg5]
      %11 = affine.min affine_map<()[s0] -> (192, s0)>()[%arg4]
      %12 = hivm.hir.mmadL1 {b_transpose, cube_producer_to_fuse_0, fixpipe_already_inserted = true} ins(%0, %inserted_slice, %true, %9, %11, %10 : tensor<256x192xbf16>, tensor<128x192xbf16>, i1, index, index, index) outs(%extracted_slice_4 : tensor<256x128xf32>) -> tensor<256x128xf32>
      %subview_5 = memref.subview %collapse_shape[0, %arg8] [256, 128] [1, 1] : memref<256x256xf32, strided<[256, 1], offset: ?>> to memref<256x128xf32, strided<[256, 1], offset: ?>>
      hivm.hir.fixpipe {enable_nz2nd, op_to_tile_0_0} ins(%12 : tensor<256x128xf32>) outs(%subview_5 : memref<256x128xf32, strided<[256, 1], offset: ?>>)
    }
  } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
  return
}