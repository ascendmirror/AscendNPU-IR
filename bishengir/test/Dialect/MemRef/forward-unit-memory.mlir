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
// CHECK-LABEL: @scalar_sqrt_forward(
// CHECK: %[[VAL_2:.*]] = memref.reinterpret_cast
// CHECK: hivm.hir.vln ins(%[[VAL_2]] : memref<1xf32, strided<[1], offset: ?>>)
// CHECK: hivm.hir.vexp ins(%[[VAL_2]] : memref<1xf32, strided<[1], offset: ?>>)
// CHECK: return
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

// -----

// CHECK-LABEL: @fused_sigmoid_gating_delta_rule_update_kernel(
#map = affine_map<()[s0] -> (128, s0)>
#map1 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map2 = affine_map<()[s0, s1] -> (s0 - s1)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1)>
#map4 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map5 = affine_map<()[s0, s1] -> (64, s0 - s1)>
module {
  func.func @fused_sigmoid_gating_delta_rule_update_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: f32, %arg7: f32, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg14: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg15: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg16: f32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, true, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<21xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i64 = arith.constant 4 : i64
    %c128_i64 = arith.constant 128 : i64
    %c8_i64 = arith.constant 8 : i64
    %c131072_i32 = arith.constant 131072 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 9.99999997E-7 : f32
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg18, %arg19 : i32
    %1 = arith.muli %0, %arg20 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.remsi %3, %arg20 : i32
    %5 = arith.divsi %3, %arg20 : i32
    %6 = arith.remsi %5, %arg19 : i32
    %7 = arith.muli %arg20, %arg19 : i32
    %8 = arith.divsi %3, %7 : i32
    %9 = arith.remsi %8, %arg18 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
    hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc : memref<128x64xf32>)
    %10 = arith.index_cast %6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg15 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1xi32>
    hivm.hir.load ins(%reinterpret_cast : memref<1xi32, strided<[1], offset: ?>>) outs(%alloc_2 : memref<1xi32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %11 = memref.load %alloc_2[%c0] : memref<1xi32>
    %12 = arith.extsi %11 : i32 to i64
    %13 = arith.muli %9, %c64_i32 : i32
    %14 = arith.muli %4, %c2_i32 : i32
    %15 = arith.muli %12, %c4_i64 : i64
    %16 = arith.muli %12, %c8_i64 : i64
    %17 = arith.index_cast %16 : i64 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg14 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    memref.store %cst, %alloc_4[%c0] : memref<1xf32>
    %18 = arith.divf %cst, %arg6 : f32
    %19 = arith.index_cast %13 : i32 to index
    %20 = affine.max #map()[%19]
    %21 = affine.min #map1()[%20, %19]
    %22 = affine.apply #map2()[%21, %19]
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    memref.store %arg6, %alloc_5[%c0] : memref<1xf32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    memref.store %18, %alloc_6[%c0] : memref<1xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    memref.store %cst_0, %alloc_7[%c0] : memref<1xf32>
    scf.for %arg21 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %23 = arith.addi %14, %arg21 : i32
      %24 = arith.divsi %23, %c2_i32 : i32
      %25 = arith.extsi %24 : i32 to i64
      %26 = arith.addi %15, %25 : i64
      %27 = arith.muli %26, %c128_i64 : i64
      %28 = arith.index_cast %27 : i64 to index
      %29 = arith.extsi %23 : i32 to i64
      %30 = arith.addi %16, %29 : i64
      %31 = arith.muli %30, %c128_i64 : i64
      %32 = arith.index_cast %31 : i64 to index
      %33 = affine.apply #map3()[%32, %19]
      %34 = arith.index_cast %23 : i32 to index
      %35 = affine.apply #map3()[%17, %34]
      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%34], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
      %reinterpret_cast_10 = memref.reinterpret_cast %arg8 to offset: [%28], sizes: [128], strides: [1] : memref<?xbf16> to memref<128xbf16, strided<[1], offset: ?>>
      %alloc_11 = memref.alloc() : memref<128xbf16>
      hivm.hir.load ins(%reinterpret_cast_10 : memref<128xbf16, strided<[1], offset: ?>>) outs(%alloc_11 : memref<128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
      hivm.hir.vcast ins(%alloc_11 : memref<128xbf16>) outs(%alloc_12 : memref<128xf32>)
      %reinterpret_cast_13 = memref.reinterpret_cast %arg9 to offset: [%28], sizes: [128], strides: [1] : memref<?xbf16> to memref<128xbf16, strided<[1], offset: ?>>
      %alloc_14 = memref.alloc() : memref<128xbf16>
      hivm.hir.load ins(%reinterpret_cast_13 : memref<128xbf16, strided<[1], offset: ?>>) outs(%alloc_14 : memref<128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
      hivm.hir.vcast ins(%alloc_14 : memref<128xbf16>) outs(%alloc_15 : memref<128xf32>)
      %reinterpret_cast_16 = memref.reinterpret_cast %arg10 to offset: [%33], sizes: [64], strides: [1] : memref<?xbf16> to memref<64xbf16, strided<[1], offset: ?>>
      %alloc_17 = memref.alloc() : memref<64xbf16>
      %subview = memref.subview %reinterpret_cast_16[0] [%22] [1] : memref<64xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      %subview_18 = memref.subview %alloc_17[0] [%22] [1] : memref<64xbf16> to memref<?xbf16, strided<[1]>>
      hivm.hir.load ins(%subview : memref<?xbf16, strided<[1], offset: ?>>) outs(%subview_18 : memref<?xbf16, strided<[1]>>) left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
      hivm.hir.vcast ins(%alloc_17 : memref<64xbf16>) outs(%alloc_19 : memref<64xf32>)
      %reinterpret_cast_20 = memref.reinterpret_cast %arg11 to offset: [%35], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
      %alloc_21 = memref.alloc() : memref<1xbf16>
      hivm.hir.load ins(%reinterpret_cast_20 : memref<1xbf16, strided<[1], offset: ?>>) outs(%alloc_21 : memref<1xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %36 = memref.load %alloc_21[%c0] : memref<1xbf16>
      %alloc_22 = memref.alloc() : memref<1xbf16>
      memref.store %36, %alloc_22[%c0] : memref<1xbf16>
      %alloc_23 = memref.alloc() : memref<1xf32>
      hivm.hir.vcast ins(%alloc_22 : memref<1xbf16>) outs(%alloc_23 : memref<1xf32>)
      %37 = memref.load %alloc_23[%c0] : memref<1xf32>
      %alloc_24 = memref.alloc() : memref<1xf32>
      hivm.hir.load ins(%reinterpret_cast_8 : memref<1xf32, strided<[1], offset: ?>>) outs(%alloc_24 : memref<1xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %38 = memref.load %alloc_24[%c0] : memref<1xf32>
      %reinterpret_cast_25 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
      %alloc_26 = memref.alloc() : memref<1xbf16>
      hivm.hir.load ins(%reinterpret_cast_25 : memref<1xbf16, strided<[1], offset: ?>>) outs(%alloc_26 : memref<1xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %39 = memref.load %alloc_26[%c0] : memref<1xbf16>
      %alloc_27 = memref.alloc() : memref<1xbf16>
      memref.store %39, %alloc_27[%c0] : memref<1xbf16>
      %alloc_28 = memref.alloc() : memref<1xf32>
      hivm.hir.vcast ins(%alloc_27 : memref<1xbf16>) outs(%alloc_28 : memref<1xf32>)
      %40 = memref.load %alloc_28[%c0] : memref<1xf32>
      %alloc_29 = memref.alloc() : memref<1xbf16>
      hivm.hir.load ins(%reinterpret_cast_9 : memref<1xbf16, strided<[1], offset: ?>>) outs(%alloc_29 : memref<1xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %41 = memref.load %alloc_29[%c0] : memref<1xbf16>
      %alloc_30 = memref.alloc() : memref<1xbf16>
      memref.store %41, %alloc_30[%c0] : memref<1xbf16>
      %alloc_31 = memref.alloc() : memref<1xf32>
      hivm.hir.vcast ins(%alloc_30 : memref<1xbf16>) outs(%alloc_31 : memref<1xf32>)
      %42 = memref.load %alloc_31[%c0] : memref<1xf32>
      %alloc_32 = memref.alloc() : memref<1xi32>
      hivm.hir.load ins(%reinterpret_cast_3 : memref<1xi32, strided<[1], offset: ?>>) outs(%alloc_32 : memref<1xi32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %43 = memref.load %alloc_32[%c0] : memref<1xi32>
      %44 = arith.cmpi sge, %43, %c0_i32 : i32
      %45 = scf.if %44 -> (memref<128x64xf32>) {
        %72 = arith.muli %43, %c131072_i32 : i32
        %73 = arith.index_cast %72 : i32 to index
        %74 = arith.muli %23, %c16384_i32 : i32
        %75 = arith.index_cast %74 : i32 to index
        %76 = affine.apply #map4()[%19, %73, %75]
        %reinterpret_cast_66 = memref.reinterpret_cast %arg13 to offset: [%76], sizes: [128, 64], strides: [128, 1] : memref<?xf32> to memref<128x64xf32, strided<[128, 1], offset: ?>>
        %alloc_67 = memref.alloc() : memref<128x64xf32>
        %77 = affine.min #map5()[%21, %19]
        %subview_68 = memref.subview %reinterpret_cast_66[0, 0] [128, %77] [1, 1] : memref<128x64xf32, strided<[128, 1], offset: ?>> to memref<128x?xf32, strided<[128, 1], offset: ?>>
        %subview_69 = memref.subview %alloc_67[0, 0] [128, %77] [1, 1] : memref<128x64xf32> to memref<128x?xf32, strided<[64, 1]>>
        hivm.hir.load ins(%subview_68 : memref<128x?xf32, strided<[128, 1], offset: ?>>) outs(%subview_69 : memref<128x?xf32, strided<[64, 1]>>) left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        scf.yield %alloc_67 : memref<128x64xf32>
      } else {
        scf.yield %alloc : memref<128x64xf32>
      }
      %46 = arith.addf %40, %42 : f32
      %47 = memref.load %alloc_5[%c0] : memref<1xf32>
      %48 = arith.mulf %47, %46 : f32
      %49 = arith.cmpf ole, %48, %arg7 : f32
      %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      memref.store %48, %alloc_33[%c0] : memref<1xf32>
      %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      hivm.hir.vexp ins(%alloc_33 : memref<1xf32>) outs(%alloc_34 : memref<1xf32>)
      %50 = memref.load %alloc_34[%c0] : memref<1xf32>
      %51 = arith.addf %50, %cst : f32
      %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      memref.store %51, %alloc_35[%c0] : memref<1xf32>
      %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      hivm.hir.vln ins(%alloc_35 : memref<1xf32>) outs(%alloc_36 : memref<1xf32>)
      %52 = memref.load %alloc_36[%c0] : memref<1xf32>
      %53 = memref.load %alloc_6[%c0] : memref<1xf32>
      %54 = arith.mulf %53, %52 : f32
      %55 = arith.select %49, %54, %46 : f32
      %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      memref.store %38, %alloc_37[%c0] : memref<1xf32>
      %alloc_38 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      hivm.hir.vexp ins(%alloc_37 : memref<1xf32>) outs(%alloc_38 : memref<1xf32>)
      %56 = memref.load %alloc_38[%c0] : memref<1xf32>
      %57 = memref.load %alloc_7[%c0] : memref<1xf32>
      %58 = arith.subf %57, %56 : f32
      %59 = arith.mulf %58, %55 : f32
      %60 = arith.subf %57, %37 : f32
      %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      memref.store %60, %alloc_39[%c0] : memref<1xf32>
      %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      hivm.hir.vexp ins(%alloc_39 : memref<1xf32>) outs(%alloc_40 : memref<1xf32>)
      %61 = memref.load %alloc_40[%c0] : memref<1xf32>
      %62 = arith.addf %61, %cst : f32
      %63 = memref.load %alloc_4[%c0] : memref<1xf32>
      %64 = arith.divf %63, %62 : f32
      %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
      hivm.hir.vmul ins(%alloc_12, %alloc_12 : memref<128xf32>, memref<128xf32>) outs(%alloc_41 : memref<128xf32>)
      %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      hivm.hir.vreduce <sum> ins(%alloc_41 : memref<128xf32>) outs(%alloc_42 : memref<1xf32>) reduce_dims = [0]
      %65 = memref.load %alloc_42[%c0] : memref<1xf32>
      %66 = math.sqrt %65 : f32
      %67 = arith.addf %66, %cst_1 : f32
      %alloc_43 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
      hivm.hir.vdiv ins(%alloc_12, %67 : memref<128xf32>, f32) outs(%alloc_43 : memref<128xf32>)
      %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
      hivm.hir.vmul ins(%alloc_15, %alloc_15 : memref<128xf32>, memref<128xf32>) outs(%alloc_44 : memref<128xf32>)
      %alloc_45 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      hivm.hir.vreduce <sum> ins(%alloc_44 : memref<128xf32>) outs(%alloc_45 : memref<1xf32>) reduce_dims = [0]
      %68 = memref.load %alloc_45[%c0] : memref<1xf32>
      %69 = math.sqrt %68 : f32
      %70 = arith.addf %69, %cst_1 : f32
      %alloc_46 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
      hivm.hir.vdiv ins(%alloc_15, %70 : memref<128xf32>, f32) outs(%alloc_46 : memref<128xf32>)
      %alloc_47 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
      hivm.hir.vmul ins(%alloc_43, %arg16 : memref<128xf32>, f32) outs(%alloc_47 : memref<128xf32>)
      %alloc_48 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      memref.store %59, %alloc_48[%c0] : memref<1xf32>
      %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      hivm.hir.vexp ins(%alloc_48 : memref<1xf32>) outs(%alloc_49 : memref<1xf32>)
      %71 = memref.load %alloc_49[%c0] : memref<1xf32>
      %alloc_50 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      hivm.hir.vmul ins(%45, %71 : memref<128x64xf32>, f32) outs(%alloc_50 : memref<128x64xf32>)
      %expand_shape = memref.expand_shape %alloc_46 [[0, 1]] output_shape [128, 1] : memref<128xf32> into memref<128x1xf32>
      %alloc_51 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      hivm.hir.vmul ins(%alloc_50, %expand_shape : memref<128x64xf32>, memref<128x1xf32>) outs(%alloc_51 : memref<128x64xf32>) broadcast = [1]
      %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
      hivm.hir.vreduce <sum> ins(%alloc_51 : memref<128x64xf32>) outs(%alloc_52 : memref<1x64xf32>) reduce_dims = [0]
      %collapse_shape = memref.collapse_shape %alloc_52 [[0, 1]] : memref<1x64xf32> into memref<64xf32>
      %alloc_53 = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
      hivm.hir.vsub ins(%alloc_19, %collapse_shape : memref<64xf32>, memref<64xf32>) outs(%alloc_53 : memref<64xf32>)
      %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
      hivm.hir.vmul ins(%alloc_53, %64 : memref<64xf32>, f32) outs(%alloc_54 : memref<64xf32>)
      %expand_shape_55 = memref.expand_shape %alloc_54 [[0, 1]] output_shape [1, 64] : memref<64xf32> into memref<1x64xf32>
      %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      hivm.hir.vmul ins(%expand_shape, %expand_shape_55 : memref<128x1xf32>, memref<1x64xf32>) outs(%alloc_56 : memref<128x64xf32>) broadcast = [0, 1]
      %alloc_57 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      hivm.hir.vadd ins(%alloc_50, %alloc_56 : memref<128x64xf32>, memref<128x64xf32>) outs(%alloc_57 : memref<128x64xf32>)
      %expand_shape_58 = memref.expand_shape %alloc_47 [[0, 1]] output_shape [128, 1] : memref<128xf32> into memref<128x1xf32>
      %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      hivm.hir.vmul ins(%alloc_57, %expand_shape_58 : memref<128x64xf32>, memref<128x1xf32>) outs(%alloc_59 : memref<128x64xf32>) broadcast = [1]
      %alloc_60 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
      hivm.hir.vreduce <sum> ins(%alloc_59 : memref<128x64xf32>) outs(%alloc_60 : memref<1x64xf32>) reduce_dims = [0]
      %collapse_shape_61 = memref.collapse_shape %alloc_60 [[0, 1]] : memref<1x64xf32> into memref<64xf32>
      scf.if %44 {
        %72 = arith.muli %43, %c131072_i32 : i32
        %73 = arith.index_cast %72 : i32 to index
        %74 = arith.muli %23, %c16384_i32 : i32
        %75 = arith.index_cast %74 : i32 to index
        %76 = affine.apply #map4()[%19, %73, %75]
        %reinterpret_cast_66 = memref.reinterpret_cast %arg13 to offset: [%76], sizes: [128, 64], strides: [128, 1] : memref<?xf32> to memref<128x64xf32, strided<[128, 1], offset: ?>>
        %77 = affine.min #map5()[%21, %19]
        %subview_67 = memref.subview %alloc_57[0, 0] [128, %77] [1, 1] : memref<128x64xf32> to memref<128x?xf32, strided<[64, 1]>>
        %subview_68 = memref.subview %reinterpret_cast_66[0, 0] [128, %77] [1, 1] : memref<128x64xf32, strided<[128, 1], offset: ?>> to memref<128x?xf32, strided<[128, 1], offset: ?>>
        hivm.hir.store ins(%subview_67 : memref<128x?xf32, strided<[64, 1]>>) outs(%subview_68 : memref<128x?xf32, strided<[128, 1], offset: ?>>)
      }
      %reinterpret_cast_62 = memref.reinterpret_cast %arg12 to offset: [%33], sizes: [64], strides: [1] : memref<?xbf16> to memref<64xbf16, strided<[1], offset: ?>>
      %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<64xbf16>
      hivm.hir.vcast ins(%collapse_shape_61 : memref<64xf32>) outs(%alloc_63 : memref<64xbf16>)
      %subview_64 = memref.subview %alloc_63[0] [%22] [1] : memref<64xbf16> to memref<?xbf16, strided<[1]>>
      %subview_65 = memref.subview %reinterpret_cast_62[0] [%22] [1] : memref<64xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      hivm.hir.store ins(%subview_64 : memref<?xbf16, strided<[1]>>) outs(%subview_65 : memref<?xbf16, strided<[1], offset: ?>>)
    }
    return
  }
}

