// RUN: bishengir-opt %s -tile-cube-vector-loop="tile-mix-cube-loop=4" -split-input-file | FileCheck %s --check-prefix=CHECK-CUBE
// RUN: bishengir-opt %s -tile-cube-vector-loop="tile-mix-vector-loop=4" -split-input-file | FileCheck %s --check-prefix=CHECK-VECTOR

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_cube(%lb: index, %ub: index, %gm_b: memref<?xf16>, %gm_offset: index, %gm_a: memref<?xf16>, %gm_c: memref<128x498xf32, strided<[498, 1], offset: ?>>) -> (memref<498x64xf16, strided<[?, ?], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %c498 = arith.constant 498 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    // Define A + Load A
    %gm_a_tile = memref.reinterpret_cast %gm_a to offset: [%gm_offset], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %a = memref.alloc() : memref<128x64xf16>
    hivm.hir.load ins(%gm_a_tile : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%a : memref<128x64xf16>) init_out_buffer = false
    %tensor_a = bufferization.to_tensor %a restrict writable : memref<128x64xf16>
    // Define B
    %gm_b_tile = memref.reinterpret_cast %gm_b to offset: [%gm_offset], sizes: [498, 64], strides: [64, 1] : memref<?xf16> to memref<498x64xf16, strided<[64, 1], offset: ?>>
    %gm_b_tile_cast = memref.cast %gm_b_tile : memref<498x64xf16, strided<[64, 1], offset: ?>> to memref<498x64xf16, strided<[?, ?], offset: ?>>
    // CHECK-CUBE: scf.for
    %res = scf.for %arg0 = %lb to %ub step %c1 iter_args(%arg1 = %gm_b_tile_cast) -> (memref<498x64xf16, strided<[?, ?], offset: ?>>) {
      // CHECK-CUBE-NOT: memref.alloc
      // CHECK-CUBE-NOT: hivm.hir.load
      // CHECK-CUBE: scf.for
      // CHECK-CUBE: memref.alloc
      // CHECK-CUBE: hivm.hir.load

      // Load B
      %b = memref.alloc() : memref<498x64xf16>
      hivm.hir.load ins(%arg1 : memref<498x64xf16, strided<[?, ?], offset: ?>>) outs(%b : memref<498x64xf16>) init_out_buffer = false
      %tensor_b = bufferization.to_tensor %b restrict writable : memref<498x64xf16>
      // Mmad + fixpipe
      %c = tensor.empty() : tensor<128x498xf32>
      %mmad_out = hivm.hir.mmadL1 {b_transpose} ins(%tensor_a, %tensor_b, %true, %c128, %c64, %c498 : tensor<128x64xf16>, tensor<498x64xf16>, i1, index, index, index) outs(%c : tensor<128x498xf32>) -> tensor<128x498xf32>
      hivm.hir.fixpipe {enable_nz2nd} ins(%mmad_out : tensor<128x498xf32>) outs(%gm_c : memref<128x498xf32, strided<[498, 1], offset: ?>>)
      // Advacne B
      %advance_b = memref.reinterpret_cast %gm_b to offset: [%gm_offset], sizes: [498, 64], strides: [64, 1] : memref<?xf16> to memref<498x64xf16, strided<[64, 1], offset: ?>>
      %advance_b_cast = memref.cast %advance_b : memref<498x64xf16, strided<[64, 1], offset: ?>> to memref<498x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %advance_b_cast : memref<498x64xf16, strided<[?, ?], offset: ?>>
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return %res : memref<498x64xf16, strided<[?, ?], offset: ?>>
  }
}

func.func @test_cube(%lb: index, %ub: index, %gm_b: memref<?xf16>, %gm_offset: index, %gm_a: memref<?xf16>, %gm_c: memref<128x498xf32, strided<[498, 1], offset: ?>>) -> (memref<498x64xf16, strided<[?, ?], offset: ?>>) {

scf.for %arg25 = %c0_14 to %83 step %c1_15 {
    %reinterpret_cast_22 = memref.reinterpret_cast %arg4 to offset: [%108], sizes: [256, 192], strides: [192, 1] : memref<?xbf16> to memref<256x192xbf16, strided<[192, 1], offset: ?>>
    %alloc_23 = memref.alloc() : memref<256x192xbf16>
    %subview_24 = memref.subview %reinterpret_cast_22[0, 0] [%102, 192] [1, 1] : memref<256x192xbf16, strided<[192, 1], offset: ?>> to memref<?x192xbf16, strided<[192, 1], offset: ?>>
    %subview_25 = memref.subview %alloc_23[0, 0] [%102, 192] [1, 1] : memref<256x192xbf16> to memref<?x192xbf16, strided<[192, 1]>>
    hivm.hir.load ins(%subview_24 : memref<?x192xbf16, strided<[192, 1], offset: ?>>) outs(%subview_25 : memref<?x192xbf16, strided<[192, 1]>>) pad_mode = <PadValue> pad_value = %cst_0 : bf16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %103 : i1 may_implicit_transpose_with_last_axis = false
    %109 = bufferization.to_tensor %alloc_23 restrict writable : memref<256x192xbf16>
    %110 = tensor.empty() : tensor<256x256xf32>
    %111 = hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%50, %109, %true, %48, %c192, %102 : tensor<256x192xbf16>, tensor<256x192xbf16>, i1, index, index, index) outs(%110 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %subview_26 = memref.subview %79[%arg25, 0, 0] [1, 256, 256] [1, 1, 1] : memref<2x256x256xf32> to memref<1x256x256xf32, strided<[65536, 256, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview_26 [[0, 1], [2]] : memref<1x256x256xf32, strided<[65536, 256, 1], offset: ?>> into memref<256x256xf32, strided<[256, 1], offset: ?>>
    hivm.hir.fixpipe {enable_nz2nd} ins(%111 : tensor<256x256xf32>) outs(%collapse_shape : memref<256x256xf32, strided<[256, 1], offset: ?>>)
    scf.yield
} {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
return
}
// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_cube(%lb: index, %ub: index, %gm_b: memref<?xf16>, %gm_offset: index, %gm_a: memref<?xf16>, %gm_c: memref<128x498xf32, strided<[498, 1], offset: ?>>) -> (memref<498x64xf16, strided<[?, ?], offset: ?>>, memref<128x64xf16, strided<[?, ?], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %c498 = arith.constant 498 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    // Define A
    %gm_a_tile = memref.reinterpret_cast %gm_a to offset: [%gm_offset], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %gm_a_tile_cast = memref.cast %gm_a_tile : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16, strided<[?, ?], offset: ?>>
    // Define B
    %gm_b_tile = memref.reinterpret_cast %gm_b to offset: [%gm_offset], sizes: [498, 64], strides: [64, 1] : memref<?xf16> to memref<498x64xf16, strided<[64, 1], offset: ?>>
    %gm_b_tile_cast = memref.cast %gm_b_tile : memref<498x64xf16, strided<[64, 1], offset: ?>> to memref<498x64xf16, strided<[?, ?], offset: ?>>
    // CHECK-CUBE: scf.for
    %res:2 = scf.for %arg0 = %lb to %ub step %c1 iter_args(%arg1 = %gm_b_tile_cast, %arg2 = %gm_a_tile_cast) -> (memref<498x64xf16, strided<[?, ?], offset: ?>>, memref<128x64xf16, strided<[?, ?], offset: ?>>) {
      // CHECK-CUBE-NOT: hivm.hir.load
      // CHECK-CUBE: scf.for
      // CHECK-CUBE: hivm.hir.load
      // CHECK-CUBE: hivm.hir.load

      // Load A
      %a = memref.alloc() : memref<128x64xf16>
      hivm.hir.load ins(%arg2 : memref<128x64xf16, strided<[?, ?], offset: ?>>) outs(%a : memref<128x64xf16>) init_out_buffer = false
      %tensor_a = bufferization.to_tensor %a restrict writable : memref<128x64xf16>
      // Load B
      %b = memref.alloc() : memref<498x64xf16>
      hivm.hir.load ins(%arg1 : memref<498x64xf16, strided<[?, ?], offset: ?>>) outs(%b : memref<498x64xf16>) init_out_buffer = false
      %tensor_b = bufferization.to_tensor %b restrict writable : memref<498x64xf16>
      // Mmad + fixpipe
      %c = tensor.empty() : tensor<128x498xf32>
      %mmad_out = hivm.hir.mmadL1 {b_transpose} ins(%tensor_a, %tensor_b, %true, %c128, %c64, %c498 : tensor<128x64xf16>, tensor<498x64xf16>, i1, index, index, index) outs(%c : tensor<128x498xf32>) -> tensor<128x498xf32>
      hivm.hir.fixpipe {enable_nz2nd} ins(%mmad_out : tensor<128x498xf32>) outs(%gm_c : memref<128x498xf32, strided<[498, 1], offset: ?>>)
      // Advacne A
      %advance_a = memref.reinterpret_cast %gm_a to offset: [%gm_offset], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %advance_a_cast = memref.cast %advance_a : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16, strided<[?, ?], offset: ?>>
      // Advacne B
      %advance_b = memref.reinterpret_cast %gm_b to offset: [%gm_offset], sizes: [498, 64], strides: [64, 1] : memref<?xf16> to memref<498x64xf16, strided<[64, 1], offset: ?>>
      %advance_b_cast = memref.cast %advance_b : memref<498x64xf16, strided<[64, 1], offset: ?>> to memref<498x64xf16, strided<[?, ?], offset: ?>>
      scf.yield %advance_b_cast, %advance_a_cast: memref<498x64xf16, strided<[?, ?], offset: ?>>, memref<128x64xf16, strided<[?, ?], offset: ?>>
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    return %res#0, %res#1 : memref<498x64xf16, strided<[?, ?], offset: ?>>, memref<128x64xf16, strided<[?, ?], offset: ?>>
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @test_cube(%lb: index, %ub: index, %gm_b: memref<?xf16>, %gm_offset: index, %gm_src: tensor<4x128x498xf32>, %gm_dst: memref<4x128x498xf16>) -> (tensor<128xf32>) {
    %c1 = arith.constant 1 : index
    %accum = tensor.empty() : tensor<128xf32>
    // CHECK-VECTOR: scf.for
    %res = scf.for %arg0 = %lb to %ub step %c1 iter_args(%arg1 = %accum) -> (tensor<128xf32>) {
      // CHECK-VECTOR: scf.for
      // CHECK-VECTOR: hivm.hir.load
      %empty = tensor.empty() : tensor<128x498xf32>
      %extracted_slice = tensor.extract_slice %gm_src[%gm_offset, 0, 0] [1, 128, 498] [1, 1, 1] : tensor<4x128x498xf32> to tensor<128x498xf32>
      %load = hivm.hir.load ins(%extracted_slice : tensor<128x498xf32>) outs(%empty : tensor<128x498xf32>) init_out_buffer = false -> tensor<128x498xf32>
      %empty1 = tensor.empty() : tensor<128x1xf32>
      %reduced = hivm.hir.vreduce <max> ins(%load : tensor<128x498xf32>) outs(%empty1 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
      %collapsed = tensor.collapse_shape %reduced [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
      %empty2 = tensor.empty() : tensor<128x498xf16>
      // tensor yields
      %cast = hivm.hir.vcast ins(%load : tensor<128x498xf32>) outs(%empty2 : tensor<128x498xf16>) -> tensor<128x498xf16>
      %subview = memref.subview %gm_dst[%gm_offset, 0, 0] [1, 128, 498] [1, 1, 1] : memref<4x128x498xf16> to memref<1x128x498xf16, strided<[63744, 498, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x498xf16, strided<[63744, 498, 1], offset: ?>> into memref<128x498xf16, strided<[498, 1], offset: ?>>
      // memref store
      hivm.hir.store ins(%cast : tensor<128x498xf16>) outs(%collapse_shape : memref<128x498xf16, strided<[498, 1], offset: ?>>)
      scf.yield %collapsed : tensor<128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    return %res : tensor<128xf32>
  }
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>>>} {
  func.func @nop(%gm_offset: index, %gm_a: memref<?xf16>) {
    %c1 = arith.constant 1 : index
    %c498 = arith.constant 498 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %gm_a_tile = memref.reinterpret_cast %gm_a to offset: [%gm_offset], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %a = memref.alloc() : memref<128x64xf16>
    // CHECK-CUBE: hivm.hir.load
    // CHECK-VECTOR: hivm.hir.load
    hivm.hir.load ins(%gm_a_tile : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%a : memref<128x64xf16>) init_out_buffer = false
    return
  }
}
