//===- HIVMOps.td - HIVM dialect operation definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for HIVM dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// HIVM Dialect operations.
//===----------------------------------------------------------------------===//

def GetBlockIdxOp : HIVM_Op<"get_block_idx", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get block idx of the current device thread used for parallelization.";
  let description = [{
    This op gets the block idx of the current device thread.
    This op will be lowered to `GetBlockIdxInstrOp`.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetBlockNumOp : HIVM_Op<"get_block_num", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get block number of the current device thread used for parallelization.";
  let description = [{
    This op gets the block number of the current device thread.
    This op will be lowered to `GetBlockNumInstrOp`.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetSubBlockIdxOp : HIVM_Op<"get_sub_block_idx", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get sub block idx of the current device thread used for parallelization.";
  let description = [{
    This op gets the sub block idx of the current device thread.
    This op will be lowered to GetSubBlockIdxInstrOp.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def GetSubBlockNumOp : HIVM_Op<"get_sub_block_num", [Pure, CubeVectorCoreTypeTrait]> {
  let summary = "Get sub block number of the current device thread used for parallelization.";
  let description = [{
    This op gets the sub block number of the current device thread.
    This op will be lowered to GetSubBlockNumInstrOp.
  }];
  let results = (outs I64:$result);
  let assemblyFormat = "attr-dict `->` type($result)";
}

def PointerCastOp : HIVM_Op<"pointer_cast",
  [CubeVectorCoreTypeTrait, AttrSizedOperandSegments]> {
  let summary = "HIVM pointer cast op at specific i64 addr";
  let description = [{
    The specific i64 addrs are stored in `$addrs`, which is variadic.

    Constraints:
    1. The type of each address should be i64.
    2. addrs should have at least one addr.

    Examples:
    ```mlir
    %addr = arith.constant 1234 : i64
    %tmp = hivm.hir.pointer_cast(%addr) : memref<32xf32>

    %addr2 = arith.constant 1600 : i64
    %addr3 = arith.constant 3200 : i64
    %tmp2 = hivm.hir.pointer_cast(%addr, %addr2) : memref<32xf32>
    %tmp3 = hivm.hir.pointer_cast(%addr, %addr2, %addr3) : memref<32xf32>
    ```
  }];
  let arguments = (ins Variadic<I64>:$addrs,
                       Variadic<Index>:$dynamicSizes
  );
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `(`$addrs `)` (`[` $dynamicSizes^`]`)? attr-dict `:` type($result)
  }];
  let builders= [
    OpBuilder<(ins "Type":$result, "Value":$addr)>,
    OpBuilder<(ins "Type":$result, "ValueRange":$addrs)>,
    OpBuilder<(ins "Type":$result, "Value":$addr, "ValueRange":$dynamicSizes)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$addr, "ValueRange":$dynamicSizes)>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    TypedValue<IntegerType> getSingleAddr();
  }];
}

def BitcastOp : HIVM_Op<"bitcast",
  [AlwaysSpeculatable, Elementwise, SameOperandsAndResultShape,
   NoMemoryEffect, Pure]> {
  let summary = "Reinterprets the bits of a shaped value without changing data";
  let description = [{
    The `bitcast` operation converts a tensor/memref from one element type to another
    while preserving the underlying bit representation. The operation requires:

    1. Same shape for input and output (2x3 != 3x2)
    2. Same total bit-width (element_bitwidth * num_elements)
    3. Same memory layout/strides (for memrefs)
  }];
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $src `:` type($src) `->` type($result) attr-dict
  }];
}

def SetFFTSBaseAddrOp : HIVM_Op<"set_ffts_base_addr", [CubeVectorCoreTypeTrait]> {
  let summary = "set base addr for ffts sync mechanism.";
  let arguments = (ins I64:$ffts_base_addr);
  let assemblyFormat = [{ 
    attr-dict $ffts_base_addr
  }];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMOPS_TD
