// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect -mlir-print-op-generic %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: test
module {
  func.func @test(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                  %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16x16xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16x16xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c8192_i64) : memref<16x16x16xf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c16384_i64) : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>) outs(%0: memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vadd ins(%0, %2: memref<16x16x16xf16, #hivm.address_space<ub>>, memref<16x16x16xf16, #hivm.address_space<ub>>) outs(%0: memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%0: memref<16x16x16xf16, #hivm.address_space<ub>>) outs(%arg1: memref<16x16x16xf16, #hivm.address_space<gm>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}

// -----

module {
  func.func @test(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<16xf32, #hivm.address_space<gm>>, %arg2: memref<256xf32, #hivm.address_space<gm>>) {
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    %1 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%1 : memref<16xf32, #hivm.address_space<cbuf>>)
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%0, %1, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%2 : memref<256xf32, #hivm.address_space<cc>>)
    hivm.hir.fixpipe {enable_nz2nd} ins(%2 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    return
  }
}
